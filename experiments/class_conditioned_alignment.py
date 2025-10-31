"""
Compute class-conditioned CORAL and MMD between MNIST and USPS models
using class-average shared nuisances.
"""

import torch
from metrics.coral import coral_distance
from metrics.mmd import mmd_rbf
from tqdm import tqdm


@torch.no_grad()
def class_conditioned_alignment(
    encoder_mnist,
    decoder_mnist,
    encoder_usps,
    decoder_usps,
    loader_mnist,
    loader_usps,
    device,
    n_classes=10,
    n_samples=1000,
):
    """
    Compute per-class CORAL and MMD using class-mean shared nuisances.

    Returns:
        dict: {
            'coral_per_class': [...],
            'mmd_per_class': [...],
            'coral_avg': float,
            'mmd_avg': float
        }
    """

    # ------------------------------------------------------------
    # 1. Collect samples
    # ------------------------------------------------------------
    def collect_latents(loader, encoder, max_samples):
        zs_s, zs_n, ys = [], [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z_s, z_n = encoder(x)
            zs_s.append(z_s)
            zs_n.append(z_n)
            ys.append(y)
            if len(torch.cat(zs_s)) >= max_samples:
                break
        return (
            torch.cat(zs_s)[:max_samples],
            torch.cat(zs_n)[:max_samples],
            torch.cat(ys)[:max_samples],
        )

    z_sM, z_nM, yM = collect_latents(loader_mnist, encoder_mnist, n_samples)
    z_sU, z_nU, yU = collect_latents(loader_usps, encoder_usps, n_samples)

    results = {"coral_per_class": [], "mmd_per_class": []}

    # ------------------------------------------------------------
    # 2. Per-class shared-nuisance reconstruction + metrics
    # ------------------------------------------------------------
    for c in tqdm(range(n_classes), desc="Class-conditioned metrics"):
        # Filter by class
        idxM = (yM == c)
        idxU = (yU == c)
        if idxM.sum() == 0 or idxU.sum() == 0:
            continue

        z_sM_c, z_nM_c = z_sM[idxM], z_nM[idxM]
        z_sU_c, z_nU_c = z_sU[idxU], z_nU[idxU]

        # Per-class mean nuisances
        mean_nM_c = z_nM_c.mean(dim=0, keepdim=True)
        mean_nU_c = z_nU_c.mean(dim=0, keepdim=True)
        shared_n_c = 0.5 * (mean_nM_c + mean_nU_c)

        # Reconstruct with shared nuisance
        xM_shared = decoder_mnist(torch.cat([z_sM_c, shared_n_c.repeat(len(z_sM_c), 1)], dim=1))
        xU_shared = decoder_usps(torch.cat([z_sU_c, shared_n_c.repeat(len(z_sU_c), 1)], dim=1))

        # Flatten for metrics
        flat = lambda x: x.view(x.size(0), -1)

        coral_c = coral_distance(flat(xM_shared), flat(xU_shared))
        mmd_c = mmd_rbf(flat(xM_shared), flat(xU_shared))

        results["coral_per_class"].append(coral_c.item())
        results["mmd_per_class"].append(mmd_c.item())

    # ------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------
    results["coral_avg"] = float(torch.tensor(results["coral_per_class"]).mean())
    results["mmd_avg"] = float(torch.tensor(results["mmd_per_class"]).mean())

    print("=== Class-Conditioned Alignment (mean over classes) ===")
    for i, (c, m) in enumerate(zip(results["coral_per_class"], results["mmd_per_class"])):
        print(f"Class {i:2d}: CORAL={c:.4f}  MMD={m:.4f}")
    print(f"Avg: CORAL={results['coral_avg']:.4f}  MMD={results['mmd_avg']:.4f}")

    return results
