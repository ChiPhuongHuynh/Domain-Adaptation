import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T


class CheXpertSubsetDataset(Dataset):
    """
    Custom dataset for your downloaded CheXpert subset.
    Expects structure:
        root/
            CheXpert/
                val/
                test/
                val_labels.csv
                test_labels.csv
    """
    def __init__(self, root, split="val", pathology="Pleural Effusion", transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.pathology = pathology
        self.transform = transform

        # Load labels CSV
        csv_path = os.path.join(root, "CheXpert", f"{split}_labels.csv")
        self.df = pd.read_csv(csv_path)

        # Build full image paths
        # In val_labels.csv paths start with: CheXpert-v1.0/valid/...
        # but your actual folder is: root/CheXpert/val/...
        self.paths = []
        for p in self.df["Path"]:
            # val_paths look like: CheXpert-v1.0/valid/patientXXXXX/studyX/viewX_frontal.jpg
            # we want: root/CheXpert/val/patientXXXXX/studyX/viewX_frontal.jpg
            parts = p.split("/")
            patient_dir = parts[-3]  # patientXXXX
            study_dir = parts[-2]    # studyX
            img_name = parts[-1]     # view1_frontal.jpg

            img_path = os.path.join(root, "CheXpert", split, patient_dir, study_dir, img_name)
            self.paths.append(img_path)

        # Find column index for pathology
        if pathology not in self.df.columns:
            raise ValueError(f"Pathology '{pathology}' not found in CSV columns")

        self.label_idx = self.df.columns.get_loc(pathology)

        # Prepare view label (nuisance): AP/PA/Lateral
        if "AP/PA" in self.df.columns:
            self.views = self.df["AP/PA"].fillna("Lateral").tolist()
        else:
            self.views = ["Unknown"] * len(self.df)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]

        # Load image
        image = Image.open(img_path).convert("L")  # grayscale

        if self.transform is not None:
            image = self.transform(image)

        # Convert grayscale → 3ch since encoder expects 3 channels
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        # Main label
        label = torch.tensor(self.df.iloc[idx, self.label_idx], dtype=torch.float32)

        # Nuisance (AP/PA/Lateral)
        view_tag = self.views[idx]
        if view_tag == "AP":
            view_label = 0
        elif view_tag == "PA":
            view_label = 1
        else:  # Lateral or missing
            view_label = 2

        view_label = torch.tensor(view_label, dtype=torch.long)

        return {
            "image": image,
            "label": label,
            "view": view_label,
            "path": img_path
        }

def get_chexpert_subset_loaders(root, batch_size=4, image_size=224):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_ds = CheXpertSubsetDataset(root, split="val",
                                   pathology="Pleural Effusion",
                                   transform=transform)

    test_ds = CheXpertSubsetDataset(root, split="test",
                                    pathology="Pleural Effusion",
                                    transform=transform)

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                              shuffle=False, num_workers=4)

    return val_loader, test_loader

def get_chexpert_weighted_sampler(root, batch_size=8, image_size=224):
    """
    Create a WeightedRandomSampler to handle class imbalance.
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_ds = CheXpertSubsetDataset(root, split="val",
                                     pathology="Pleural Effusion",
                                     transform=transform)
    
    test_ds = CheXpertSubsetDataset(root, split="test",
                                    pathology="Pleural Effusion",
                                    transform=transform)

    # ---- Build sample weights for WeightedRandomSampler ----
    labels = torch.tensor(
        [val_ds.df.iloc[i][val_ds.label_idx] for i in range(len(val_ds))],
        dtype=torch.float32,
    )
    N = len(labels)
    N_pos = labels.sum()
    N_neg = N - N_pos

    # Avoid division by zero
    N_pos = max(N_pos, 1.0)
    N_neg = max(N_neg, 1.0)

    weight_pos = N / (2.0 * N_pos)
    weight_neg = N / (2.0 * N_neg)

    sample_weights = torch.where(labels == 1.0,
                                 torch.full_like(labels, weight_pos),
                                 torch.full_like(labels, weight_neg))

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=N,      # one “epoch” = N samples
        replacement=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return val_loader, test_loader