import torch.nn as nn
import torch.nn.functional as F

eps = 1e-6

class SplitEncoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64, signal_dim=32):
        super().__init__()
        self.signal_dim = signal_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.net(x)
        z_sig = z[:, :self.signal_dim]
        z_nui = z[:, self.signal_dim:]
        return z_sig, z_nui


class SplitDecoder(nn.Module):
    def __init__(self, latent_dim=64, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z)
    

class LinearProbe(nn.Module):
    def __init__(self, input_dim=32, n_classes=10): # input dim can be signal or nuisance depending on use
        super().__init__()
        self.net = nn.Linear(input_dim, n_classes)
    
    def forward(self, z_sig):
        return self.net(z_sig)
