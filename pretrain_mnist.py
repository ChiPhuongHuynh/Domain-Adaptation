"""
experiments/pretrain_usps.py

Example script to pretrain encoder/decoder/probe on the USPS dataset
using direct label supervision (no teacher model).
"""

import torch
from data.loader import get_dataloader
from models.models import SplitEncoder
from models.models import SplitDecoder
from models.models import LinearProbe
from utils.seed import set_seed
from utils.training import pretrain

set_seed(42)
# ------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# 2. Data
# ------------------------------------------------------------
train_loader = get_dataloader("mnist", batch_size=128, train=True)
test_loader  = get_dataloader("mnist", batch_size=128, train=False)

# ------------------------------------------------------------
# 3. Models
# ------------------------------------------------------------
input_dim = 784
output_dim = 784
latent_dim = 64
signal_dim = 32
num_classes = 10

encoder = SplitEncoder(input_dim=input_dim, latent_dim=latent_dim, signal_dim=signal_dim).to(device)
decoder = SplitDecoder(latent_dim=latent_dim, output_dim=output_dim).to(device)
probe   = LinearProbe(signal_dim=signal_dim, n_classes=num_classes).to(device)

# ------------------------------------------------------------
# 4. Training
# ------------------------------------------------------------
pretrain(
    encoder=encoder,
    decoder=decoder,
    probe=probe,
    dataloader=train_loader,
    device=device,
    lambda_cls=1.0,
    lambda_rec=0.5,
    lambda_preserve=1.0,
    lambda_cov=0.5,
    lambda_proj=0.1,
    lambda_cov_cycle=0.5,
    lambda_cycle_nuisance=1.0,
    epochs=20,
    save_path="artifacts/mnist/mnist_pretrained.pt"
)

# ------------------------------------------------------------
# 5. Optional: evaluate probe accuracy on test set
# ------------------------------------------------------------
encoder.eval()
probe.eval()
correct = total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        z_s, _ = encoder(x)
        logits = probe(z_s)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"✅ USPS Probe accuracy on test set: {100*correct/total:.2f}%")