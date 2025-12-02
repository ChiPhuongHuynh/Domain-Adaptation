import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from models.mimic_cxr.models import CXREncoder, LatentSplitter
from data.chexpert_subset import get_chexpert_weighted_sampler

device = torch.device("cpu")

# ----------------------------------------------------------------------------
# 1. Load Data
# ----------------------------------------------------------------------------
root = "./chexlocalize_download"
train_loader, test_loader = get_chexpert_weighted_sampler(root, batch_size=8)

# ----------------------------------------------------------------------------
# 2. Load Encoder + Splitter (frozen)
# ----------------------------------------------------------------------------
encoder = CXREncoder(backbone_name="resnet18", pretrained=True, latent_dim=1024)
splitter = LatentSplitter(latent_dim=1024, split_dim=512)

encoder.to(device)
splitter.to(device)

encoder.eval()   # freeze encoder
splitter.eval()

# ----------------------------------------------------------------------------
# 3. Linear Probe
# ----------------------------------------------------------------------------
probe = nn.Linear(512, 1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(probe.parameters(), lr=1e-3)

# ----------------------------------------------------------------------------
# 4. Train probe for a few epochs (CPU-friendly)
# ----------------------------------------------------------------------------
for epoch in range(10):
    probe.train()
    total_loss = 0

    for batch in train_loader:
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        with torch.no_grad():
            z = encoder(imgs)
            z_sig, z_nui = splitter(z)

        logits = probe(z_sig)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# ----------------------------------------------------------------------------
# 5. Evaluate probe (AUC)
# ----------------------------------------------------------------------------
probe.eval()
preds = []
targets = []

with torch.no_grad():
    for batch in test_loader:
        imgs = batch["image"].to(device)
        labels = batch["label"].numpy().tolist()

        z = encoder(imgs)
        z_sig, _ = splitter(z)

        logits = probe(z_sig)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()

        preds.extend(probs)
        targets.extend(labels)

auc = roc_auc_score(targets, preds)
print("Test AUC:", auc)
