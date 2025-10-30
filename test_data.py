# extract_latents.py
import torch
from data.loader import get_dataloader
from data.utils import save_latents

device = "cuda" if torch.cuda.is_available() else "cpu"

mnist_loader = get_dataloader("usps", batch_size=128, train=False)
