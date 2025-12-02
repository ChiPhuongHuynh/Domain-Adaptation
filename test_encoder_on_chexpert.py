import torch
from models.mimic_cxr.models import CXREncoder, LatentSplitter
from data.chexpert_subset import get_chexpert_subset_loaders

device = torch.device("cpu")

# Load data
root = "chexlocalize_download"
val_loader, _ = get_chexpert_subset_loaders(root, batch_size=2)

# Load encoder + splitter
encoder = CXREncoder(backbone_name="resnet18", pretrained=False, latent_dim=1024).to(device)
splitter = LatentSplitter(latent_dim=1024, split_dim=512).to(device)

batch = next(iter(val_loader))
images = batch["image"].to(device)

with torch.no_grad():
    z = encoder(images)
    z_sig, z_nui = splitter(z)

print("Input:", images.shape)
print("z:", z.shape)
print("z_sig:", z_sig.shape)
print("z_nui:", z_nui.shape)
