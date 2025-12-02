"""Testing individual components of the mimic_cxr project."""

import torch
import os, sys

# Get project root (parent of mimic_cxr_test)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Detected project root:", ROOT)

# Insert it into sys.path
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
print("Updated sys.path:", sys.path)

from models.mimic_cxr.models import CXREncoder, LatentSplitter

def test_cxr_encoder_forward():
    batch_size = 4
    H, W = 320, 320

    # Fake grayscale CXR batch
    x = torch.randn(batch_size, 1, H, W)

    encoder = CXREncoder(
        backbone_name="resnet18",  # faster for tests
        pretrained=False,
        latent_dim=1024,
        pool="avg",
    )

    splitter = LatentSplitter(latent_dim=1024, split_dim=512)

    with torch.no_grad():
        z = encoder(x)
        z_sig, z_nui = splitter(z)

    print("Input shape:", x.shape)
    print("z shape:", z.shape)
    print("z_sig shape:", z_sig.shape)
    print("z_nui shape:", z_nui.shape)

    assert z.shape == (batch_size, 1024)
    assert z_sig.shape == (batch_size, 512)
    assert z_nui.shape == (batch_size, 512)
    print("✅ Encoder + Splitter forward pass OK")

if __name__ == "__main__":
    test_cxr_encoder_forward()