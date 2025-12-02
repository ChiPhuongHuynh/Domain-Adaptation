import torch
import torch.nn as nn
import torchvision.models as models


class CXREncoder(nn.Module):
    """
    Image encoder for MIMIC-CXR.
    - Uses a torchvision backbone (default: ResNet-50)
    - Outputs a global latent vector z of dimension latent_dim
    """
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = False,
        latent_dim: int = 1024,
        pool: str = "avg",
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.latent_dim = latent_dim

        # ---- 1. Build backbone ----
        if backbone_name == "resnet50":
            backbone = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
            in_dim = backbone.fc.in_features
            # remove classifier head
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # up to conv5_x
        elif backbone_name == "resnet18":
            backbone = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
            in_dim = backbone.fc.in_features
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # ---- 2. Pooling ----
        if pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == "max":
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported pool type: {pool}")
        self.pool = pool

        # ---- 3. Projection to latent_dim ----
        self.projection = nn.Sequential(
            nn.Flatten(),                  # (B, C, 1, 1) -> (B, C)
            nn.Linear(in_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: (B, 1 or 3, H, W) tensor
        returns: z of shape (B, latent_dim)
        """
        # If grayscale, repeat channel to 3 for ResNet
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        feats = self.feature_extractor(x)      # (B, C, H', W')
        pooled = self.global_pool(feats)       # (B, C, 1, 1)
        z = self.projection(pooled)            # (B, latent_dim)
        return z


class LatentSplitter(nn.Module):
    """
    Simple split head:
    - Takes z (B, latent_dim)
    - Outputs z_sig and z_nui, both (B, split_dim)
    You can later swap this for something fancier; this is a clean, testable start.
    """
    def __init__(self, latent_dim: int, split_dim: int = 512):
        super().__init__()
        assert split_dim <= latent_dim, "split_dim must be <= latent_dim"

        self.latent_dim = latent_dim
        self.split_dim = split_dim

        # Simple MLP heads (project into two subspaces)
        self.to_signal = nn.Sequential(
            nn.Linear(latent_dim, split_dim),
            nn.BatchNorm1d(split_dim),
            nn.ReLU(inplace=True),
        )
        self.to_nuisance = nn.Sequential(
            nn.Linear(latent_dim, split_dim),
            nn.BatchNorm1d(split_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        """
        z: (B, latent_dim)
        returns: z_sig, z_nui both (B, split_dim)
        """
        z_sig = self.to_signal(z)
        z_nui = self.to_nuisance(z)
        return z_sig, z_nui


class CXRDecoder(nn.Module):
    """
    Decoder that reconstructs a 3x224x224 CXR image from the latent embedding.
    Input is concatenated [z_sig, z_nui] → size = 1024.
    """
    def __init__(self, latent_dim=1024):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.ReLU(True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 7→14
            nn.BatchNorm2d(256), nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 14→28
            nn.BatchNorm2d(128), nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 28→56
            nn.BatchNorm2d(64), nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 56→112
            nn.BatchNorm2d(32), nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # 112→224
            nn.Sigmoid()     # output in [0,1]
        )

    def forward(self, z_sig, z_nui):
        z = torch.cat([z_sig, z_nui], dim=1)
        h = self.fc(z)
        h = h.view(z.size(0), 512, 7, 7)
        x_recon = self.deconv(h)
        return x_recon


class CXRModel(nn.Module):
    def __init__(self, latent_dim=1024, split_dim=512, pretrained=True):
        super().__init__()
        self.encoder = CXREncoder(
            backbone_name="resnet18",
            pretrained=pretrained,
            latent_dim=latent_dim,
            pool="avg",
        )
        self.splitter = LatentSplitter(latent_dim, split_dim)
        self.decoder = CXRDecoder(latent_dim)

        # probe
        self.probe = nn.Linear(split_dim, 1) # Pleural Effusion

    def forward(self, x):
        z = self.encoder(x) # (B, latent_dim)
        z_sig, z_nui = self.splitter(z) # both (B, split_dim)
        x_recon = self.decoder(z_sig, z_nui) # (B, 3, 224, 224)
        logits = self.probe(z_sig) # (B, 1)
        return z, z_sig, z_nui, x_recon, logits
