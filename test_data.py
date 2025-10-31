# extract_latents.py
import torch
from torchvision import datasets, transforms
from data.loader import get_dataloader
from data.utils import save_latents


device = "cuda" if torch.cuda.is_available() else "cpu"

usps_loader = get_dataloader("mnist", batch_size=128, train=False)

set1 = iter(usps_loader)
img = next(set1)[0]
print(img.shape)