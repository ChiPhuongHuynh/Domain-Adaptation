import torch
from data.chexpert_subset import get_chexpert_subset_loaders

root = "chexlocalize_download"
val_loader, test_loader = get_chexpert_subset_loaders(root, batch_size=2)

batch = next(iter(val_loader))

print("Image batch shape:", batch["image"].shape)
print("Label shape:", batch["label"].shape)
print("View labels:", batch["view"])
print("First paths:", batch["path"])
