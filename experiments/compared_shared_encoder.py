"""
experiments/compare_shared_encoder.py

Evaluate cross-domain alignment using a *shared encoder/decoder* (e.g., MNIST-trained)
to isolate data-driven vs. model-driven domain gaps.
"""

import torch
import matplotlib.pyplot as plt
import os
from metrics.coral import coral_distance
from metrics.mmd import mmd_rbf


@torch.no_grad()
def compare_shared_encoder_alignment(
    encoder,
    decoder,
    loader_mnist,
    loader_usps,
    device,
    n_samples=1000,
    save_dir="artifacts/plots/shared_encoder_eval",
    visualize=True,
):
    """
    Run MNIST-trained encoder/decoder on both MNIST & USPS test sets.
    Computes CORAL + MMD across latent components and reconstructions.

    Args:
        encoder, decoder: trained on MNIST
        loader_mnist, loader_usps: DataLoaders
        device: CUDA/CPU device
        n_samples: number of samples to compare
        save_dir: directory for plots
        visualize: save sample grids
    """

    # ------------------------------------------------------------
    # 1. Collect batches
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
    # 2. Encode both domains with the same model
    # ------------------------------------------------------------
    z_sM, z_nM = encoder(xM)
    z_sU, z_nU = encoder(xU)

    with torch.no_grad():
        def stats(z_s, z_n, name):
            var_s = z_s.var(dim=0).mean().item()
            var_n = z_n.var(dim=0).mean().item()
            norm_s = z_s.norm(dim=1).mean().item()
            norm_n = z_n.norm(dim=1).mean().item()
            print(f"[{name}] Var(z_s)={var_s:.4e}  Var(z_n)={var_n:.4e}  "
                f"‖z_s‖={norm_s:.4f}  ‖z_n‖={norm_n:.4f}")
            return var_s, var_n, norm_s, norm_n

        stats(z_sM, z_nM, "MNIST")
        stats(z_sU, z_nU, "USPS")

    mean_nM = z_nM.mean(dim=0, keepdim=True)
    z_nU_stationary = mean_nM.repeat(z_nU.size(0), 1)

    # ------------------------------------------------------------
    # 3. Decode to reconstructions
    # ------------------------------------------------------------
    x_hat_M = decoder(torch.cat([z_sM, z_nM], dim=1))
    x_hat_U = decoder(torch.cat([z_sU, z_nU], dim=1))
    x_hat_U_stationary = decoder(torch.cat([z_sU, z_nU_stationary], dim=1))

    # ------------------------------------------------------------
    # 3b. Nuisance usage / swap test (MNIST domain)
    # ------------------------------------------------------------
    with torch.no_grad():
        # Original reconstructions (already have x_hat_M)
        x_orig = x_hat_M

        # Shuffle z_n within batch to break pairing
        perm = torch.randperm(z_nM.size(0), device=device)
        z_n_shuf = z_nM[perm]

        # Recon with swapped nuisance
        x_swap = decoder(torch.cat([z_sM, z_n_shuf], dim=1))

        # Mean absolute pixel-level change
        delta = (x_orig - x_swap).abs().mean().item()
        print(f"Δrecon after z_n swap (mean |x−x'|) = {delta:.6f}")


    # ------------------------------------------------------------
    # 4. Compute metrics (flattened for CORAL/MMD)
    # ------------------------------------------------------------
    flat = lambda t: t.view(t.size(0), -1)

    def evaluate(a, b, name):
        coral = coral_distance(flat(a), flat(b))
        mmd = mmd_rbf(flat(a), flat(b))
        print(f"{name:25s} | CORAL={coral:.4f} | MMD={mmd:.4f}")
        return {"name": name, "coral": coral.item(), "mmd": mmd.item()}

    print("\n=== Shared Encoder Cross-Domain Alignment ===")

    results = []
    results.append(evaluate(z_sM, z_sU, "Signal latent"))
    results.append(evaluate(z_nM, z_nU, "Nuisance latent"))
    results.append(
        evaluate(torch.cat([z_sM, z_nM], 1),
                 torch.cat([z_sU, z_nU], 1),
                 "Full latent"))
    results.append(evaluate(x_hat_M, x_hat_U, "Reconstruction"))
    results.append(evaluate(x_hat_M, x_hat_U_stationary, "Stationarized recon"))

    # ------------------------------------------------------------
    # 5. Visualization (optional)
    # ------------------------------------------------------------
    if visualize:
        os.makedirs(save_dir, exist_ok=True)

        def ensure_3d(img):
            return img.unsqueeze(0) if img.dim() == 2 else img

        def pick(imgs, n=8):
            imgs = imgs[:n].detach().cpu()
            return torch.stack([ensure_3d(im) for im in imgs])

        n = 8
        sets = {
            "MNIST orig":  pick(xM, n),
            "USPS orig":   pick(xU, n),
            "MNIST recon": pick(x_hat_M, n),
            "USPS recon":  pick(x_hat_U, n),
            "USPS stationary": pick(x_hat_U_stationary, n),
        }

        fig, axes = plt.subplots(len(sets)//2 + 1, 2, figsize=(6, 10))
        plt.subplots_adjust(hspace=0.3, wspace=0.05)

        def show_row(ax, imgs, title):
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
            grid = torch.cat([im for im in imgs], dim=-1)
            ax.imshow(grid.squeeze(), cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        for i, (name, imgs) in enumerate(sets.items()):
            r, c = divmod(i, 2)
            show_row(axes[r, c], imgs, name)

        plt.suptitle("Shared Encoder Cross-Domain Comparison", fontsize=12)
        out_path = os.path.join(save_dir, "shared_encoder_grid.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Figure saved to {out_path}")

    # ------------------------------------------------------------
    return {
        "coral": {r["name"]: r["coral"] for r in results},
        "mmd": {r["name"]: r["mmd"] for r in results},
    }
