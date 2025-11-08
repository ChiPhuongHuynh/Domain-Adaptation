import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
import os


@torch.no_grad()
def plot_latent_tsne_grid(
    encoder,
    decoder,
    dataloader,
    device,
    save_path=None,
    n_samples=1000,
    title_prefix="Latent t-SNE",
    point_size=8,
    perplexity=30,
    random_state=42,
):
    """
    Visualize signal and nuisance latent spaces before and after reconstruction.
    Produces a 2x2 grid:
        Row 1: encoder(x)
            [Left]  z_signal
            [Right] z_nuisance
        Row 2: encoder(decoder(encoder(x)))

    Args:
        encoder (nn.Module): Trained encoder returning (z_signal, z_nuisance).
        decoder (nn.Module): Trained decoder taking [z_signal, z_nuisance].
        dataloader (DataLoader): Dataset to sample (x, y) from.
        device (torch.device): CUDA or CPU.
        save_path (str, optional): Path to save plot. If None, only shows it.
        n_samples (int): Number of samples to visualize.
        title_prefix (str): Title prefix for figure.
        point_size (int): Size of t-SNE scatter points.
        perplexity (int): t-SNE perplexity.
        random_state (int): Random seed for t-SNE.
    """

    encoder.eval()
    decoder.eval()

    # ------------------------------------------------------------
    # 1. Collect samples
    # ------------------------------------------------------------
    xs, ys = [], []
    for x, y in dataloader:
        xs.append(x)
        ys.append(y)
        if len(torch.cat(xs)) >= n_samples:
            break
    x = torch.cat(xs)[:n_samples].to(device)
    y = torch.cat(ys)[:n_samples].cpu().numpy()

    # ------------------------------------------------------------
    # 2. Encode + Reconstruct + Re-encode
    # ------------------------------------------------------------
    z_s, z_n = encoder(x)
    x_hat = decoder(torch.cat([z_s, z_n], dim=1))
    z_s_rec, z_n_rec = encoder(x_hat)

    # Convert to CPU numpy for t-SNE
    z_s = z_s.cpu().numpy()
    z_n = z_n.cpu().numpy()
    z_s_rec = z_s_rec.cpu().numpy()
    z_n_rec = z_n_rec.cpu().numpy()

    # ------------------------------------------------------------
    # 3. Joint t-SNE fit for consistent scale
    # ------------------------------------------------------------
    print("[t-SNE] Running joint projection...")
    all_z = torch.cat(
        [
            torch.tensor(z_s),
            torch.tensor(z_n),
            torch.tensor(z_s_rec),
            torch.tensor(z_n_rec),
        ],
        dim=0,
    ).numpy()

    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state, init="pca"
    )
    all_z_2d = tsne.fit_transform(all_z)

    N = len(z_s)
    Z_s, Z_n = all_z_2d[:N], all_z_2d[N:2 * N]
    Z_s_rec, Z_n_rec = all_z_2d[2 * N:3 * N], all_z_2d[3 * N:4 * N]

    # ------------------------------------------------------------
    # 4. Plot 2x2 grid
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    axes[0, 0].scatter(Z_s[:, 0], Z_s[:, 1], c=y, cmap="tab10", s=point_size)
    axes[0, 0].set_title("Signal (encoder(x))")

    axes[0, 1].scatter(Z_n[:, 0], Z_n[:, 1], c=y, cmap="tab10", s=point_size)
    axes[0, 1].set_title("Nuisance (encoder(x))")

    axes[1, 0].scatter(Z_s_rec[:, 0], Z_s_rec[:, 1], c=y, cmap="tab10", s=point_size)
    axes[1, 0].set_title("Signal (encoder(decoder(x)))")

    axes[1, 1].scatter(Z_n_rec[:, 0], Z_n_rec[:, 1], c=y, cmap="tab10", s=point_size)
    axes[1, 1].set_title("Nuisance (encoder(decoder(x)))")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"{title_prefix} — Signal vs Nuisance Latents", fontsize=14)
    plt.tight_layout()

    # ------------------------------------------------------------
    # 5. Save or show
    # ------------------------------------------------------------
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ t-SNE plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

def plot_latent_tsne_before_after(
    encoder_pre,
    encoder_finetuned,
    dataloader,
    device,
    save_path=None,
    n_samples=1000,
    point_size=8,
    perplexity=30,
    random_state=42,
):
    """
    Compare t-SNE embeddings of signal and nuisance latents
    before vs. after finetuning.

    Layout:
        Row 1: Pretrained encoder (z_s, z_n)
        Row 2: Finetuned encoder (z_s, z_n)
    """

    # -------------------------
    # 1. Collect samples
    # -------------------------
    xs, ys = [], []
    for x, y in dataloader:
        xs.append(x)
        ys.append(y)
        if len(torch.cat(xs)) >= n_samples:
            break
    x = torch.cat(xs)[:n_samples].to(device)
    y = torch.cat(ys)[:n_samples].cpu().numpy()

    # -------------------------
    # 2. Encode using both encoders
    # -------------------------
    with torch.no_grad():
        z_s_pre, z_n_pre = encoder_pre(x)
        z_s_post, z_n_post = encoder_finetuned(x)

    # to CPU numpy
    z_s_pre, z_n_pre = z_s_pre.cpu().numpy(), z_n_pre.cpu().numpy()
    z_s_post, z_n_post = z_s_post.cpu().numpy(), z_n_post.cpu().numpy()

    # -------------------------
    # 3. Joint t-SNE (shared projection space)
    # -------------------------
    print("[t-SNE] Running joint projection...")
    all_z = torch.cat(
        [
            torch.tensor(z_s_pre),
            torch.tensor(z_n_pre),
            torch.tensor(z_s_post),
            torch.tensor(z_n_post),
        ],
        dim=0,
    ).numpy()

    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state, init="pca"
    )
    all_z_2d = tsne.fit_transform(all_z)

    N = len(z_s_pre)
    Z_s_pre = all_z_2d[:N]
    Z_n_pre = all_z_2d[N:2*N]
    Z_s_post = all_z_2d[2*N:3*N]
    Z_n_post = all_z_2d[3*N:4*N]

    # -------------------------
    # 4. Plot 2×2 grid
    # -------------------------
    sns.set_style("white")
    palette = palette = sns.color_palette("tab20", 10)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)

    def scatter(ax, Z, title):
        ax.scatter(Z[:,0], Z[:,1], c=[palette[i] for i in y],
                   s=point_size, alpha=0.8, linewidth=0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('white')

    scatter(axes[0,0], Z_s_pre, "Signal (Pretrain)")
    scatter(axes[0,1], Z_n_pre, "Nuisance (Pretrain)")
    scatter(axes[1,0], Z_s_post, "Signal (Finetune)")
    scatter(axes[1,1], Z_n_post, "Nuisance (Finetune)")

    fig.suptitle("t-SNE of Latent Spaces — Pretrain vs Finetune", fontsize=13)
    """
    plt.tight_layout(rect=[0,0,1,0.96])
    handles = [plt.Line2D([], [], marker='o', color=palette[i],
                      linestyle='', markersize=6)
           for i in range(10)]
    labels = [str(i) for i in range(10)]
    fig.legend(handles, labels, loc='lower center', ncol=10, frameon=False)
    """

    # -------------------------
    # 5. Save or show
    # -------------------------
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ t-SNE plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
