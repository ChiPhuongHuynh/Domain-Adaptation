"""
experiments/compare_shared_nuisance.py

Evaluate how shared nuisance latents affect cross-domain alignment
between MNIST and USPS models.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from metrics.coral import coral_distance
from metrics.mmd import mmd_rbf
import os


@torch.no_grad()
def compare_shared_nuisance_alignment(
    encoder_mnist,
    decoder_mnist,
    encoder_usps,
    decoder_usps,
    loader_mnist,
    loader_usps,
    device,
    n_samples=1000,
    save_dir="artifacts/plots/shared_nuisance_eval",
    visualize=True,
):
    """
    Steps 2–4 from your plan: shared-nuisance synthesis + cross-domain comparison.

    Args:
        encoder_mnist, decoder_mnist, encoder_usps, decoder_usps: trained models
        loader_mnist, loader_usps: DataLoaders with (x, y)
        device: CUDA/CPU device
        n_samples: number of samples per domain
        save_dir: where to save results
        visualize: if True, saves sample grid images
    """

    # ------------------------------------------------------------
    # 1. Collect MNIST and USPS batches
    # ------------------------------------------------------------
    def get_batch(loader, n):
        xs, ys = [], []
        for x, y in loader:
            xs.append(x)
            ys.append(y)
            if len(torch.cat(xs)) >= n:
                break
        return torch.cat(xs)[:n].to(device), torch.cat(ys)[:n].to(device)

    xM, yM = get_batch(loader_mnist, n_samples)
    xU, yU = get_batch(loader_usps, n_samples)

    # ------------------------------------------------------------
    # 2. Encode to latent spaces
    # ------------------------------------------------------------
    z_sM, z_nM = encoder_mnist(xM)
    z_sU, z_nU = encoder_usps(xU)

    mean_nM = z_nM.mean(dim=0, keepdim=True)
    mean_nU = z_nU.mean(dim=0, keepdim=True)
    shared_z_n = 0.5 * (mean_nM + mean_nU)

    # ------------------------------------------------------------
    # 3. Reconstructions
    # ------------------------------------------------------------
    # (a) Original reconstructions
    x_hat_M = decoder_mnist(torch.cat([z_sM, z_nM], dim=1))
    x_hat_U = decoder_usps(torch.cat([z_sU, z_nU], dim=1))

    # (b) Shared-nuisance reconstructions
    x_shared_M = decoder_mnist(torch.cat([z_sM, shared_z_n.repeat(len(z_sM), 1)], dim=1))
    x_shared_U = decoder_usps(torch.cat([z_sU, shared_z_n.repeat(len(z_sU), 1)], dim=1))

    # ------------------------------------------------------------
    # 4. Quantitative metrics
    # ------------------------------------------------------------
    # Flatten images for CORAL / MMD
    flat = lambda x: x.view(x.size(0), -1)

    coral_orig = coral_distance(flat(xM), flat(xU))
    coral_recon = coral_distance(flat(x_hat_M), flat(x_hat_U))
    coral_shared = coral_distance(flat(x_shared_M), flat(x_shared_U))

    mmd_orig = mmd_rbf(flat(xM), flat(xU))
    mmd_recon = mmd_rbf(flat(x_hat_M), flat(x_hat_U))
    mmd_shared = mmd_rbf(flat(x_shared_M), flat(x_shared_U))

    print("=== Cross-domain alignment (lower is better) ===")
    print(f"CORAL  original: {coral_orig.item():.4f}")
    print(f"CORAL  reconstr: {coral_recon.item():.4f}")
    print(f"CORAL  shared-n: {coral_shared.item():.4f}")
    print(f"MMD    original: {mmd_orig.item():.4f}")
    print(f"MMD    reconstr: {mmd_recon.item():.4f}")
    print(f"MMD    shared-n: {mmd_shared.item():.4f}")

    results = {
        "coral": {
            "original": coral_orig.item(),
            "reconstructed": coral_recon.item(),
            "shared_nuisance": coral_shared.item(),
        },
        "mmd": {
            "original": mmd_orig.item(),
            "reconstructed": mmd_recon.item(),
            "shared_nuisance": mmd_shared.item(),
        },
    }

    # ------------------------------------------------------------
    # 5. Optional visualization
    # ------------------------------------------------------------
    if visualize:
        os.makedirs(save_dir, exist_ok=True)

        # Utility: ensure every image has shape [1,H,W]
        def ensure_3d(img):
            if img.dim() == 2:
                img = img.unsqueeze(0)
            return img

        def pick_batch(imgs, n=8):
            imgs = imgs[:n].detach().cpu()
            imgs = torch.stack([ensure_3d(im) for im in imgs])  # [N,1,H,W]
            return imgs

        # Pick small subsets
        n = 8
        sets = {
            "MNIST orig":  pick_batch(xM, n),
            "USPS orig":   pick_batch(xU, n),
            "MNIST recon": pick_batch(x_hat_M, n),
            "USPS recon":  pick_batch(x_hat_U, n),
            "MNIST shared-n": pick_batch(x_shared_M, n),
            "USPS shared-n":  pick_batch(x_shared_U, n),
        }

        # 3 rows × 2 cols
        fig, axes = plt.subplots(3, 2, figsize=(6, 9))
        plt.subplots_adjust(hspace=0.25, wspace=0.05)

        def show_row(ax, imgs, title):
            # normalize for display
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
            # horizontally concatenate [1,H,W] → [1,H,n*W]
            grid = torch.cat([im for im in imgs], dim=-1)
            ax.imshow(grid.squeeze(), cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        titles = list(sets.keys())
        for i, (name, imgs) in enumerate(sets.items()):
            r, c = divmod(i, 2)
            show_row(axes[r, c], imgs, name)

        plt.suptitle("Cross-Domain Shared-Nuisance Comparison", fontsize=12)
        out_path = os.path.join(save_dir, "shared_nuisance_grid.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Combined figure saved to {out_path}")


