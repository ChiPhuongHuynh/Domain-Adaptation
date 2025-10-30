import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.losses import (
    probe_loss,
    conservative_reconstruction_loss,
    cross_covariance_norm,
    covariance_penalty,
    projection_penalty,
    normalize_batch
)

eps = 1e-6

def pretrain(
    encoder, decoder, probe, dataloader, device,
    lambda_cls=1.0, lambda_rec=0.5, lambda_preserve=1.0,
    lambda_cov=0.5, lambda_proj=0.1, lambda_cov_cycle=0.5,
    lambda_cycle_nuisance=1.0,
    epochs=20, save_path="mnist_pretrained_on_label.pt"
):
    """
    Pretraining with direct label supervision via probe, without teacher distillation.
    Combines supervised probe loss, reconstruction, and latent consistency penalties.
    """

    encoder.train()
    decoder.train()
    probe.train()

    opt = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(probe.parameters()),
        lr=1e-3
    )

    for epoch in range(epochs):
        total_loss = 0.0
        total_cls = total_rec = total_cov = 0.0

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            # -------------------
            # Forward pass
            # -------------------
            z_s, z_n = encoder(x)
            logits = probe(z_s)

            # -------------------
            # Loss components
            # -------------------

            # (1) Supervised probe classification loss
            L_cls = F.cross_entropy(logits, y)

            # (2) Reconstruction
            z_n_noisy = z_n + torch.randn_like(z_n) * 0.01
            x_hat = decoder(torch.cat([z_s.detach(), z_n_noisy], dim=1))
            L_rec, rec_stats = conservative_reconstruction_loss(
                x.view(x.size(0), -1), x_hat
            )

            # (3) Signal preservation & cycle consistency
            z_s_p, z_n_p = encoder(x_hat)
            L_preserve = torch.norm(z_s - z_s_p, dim=1).mean()
            L_cycle_nuisance = torch.norm(z_n_p - z_n, dim=1).mean()

            # (4) Cross-covariance and projection penalties
            z_s_norm = normalize_batch(z_s)
            z_n_norm = normalize_batch(z_n)

            L_cov = cross_covariance_norm(z_s_norm, z_n_norm)
            L_cov_cycle = covariance_penalty(z_s_p, z_n_p)
            L_proj = projection_penalty(z_s, z_n)

            # -------------------
            # Total loss
            # -------------------
            loss = (
                lambda_cls * L_cls
                + lambda_rec * L_rec
                + lambda_preserve * L_preserve
                + lambda_cov * L_cov
                + lambda_proj * L_proj
                + lambda_cov_cycle * L_cov_cycle
                + lambda_cycle_nuisance * L_cycle_nuisance
            )

            # -------------------
            # Backpropagation
            # -------------------
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total_cls += L_cls.item() * x.size(0)
            total_rec += L_rec.item() * x.size(0)
            total_cov += L_cov.item() * x.size(0)

        # -------------------
        # Logging
        # -------------------
        n = len(dataloader.dataset)
        print(f"[Pretrain Epoch {epoch+1}/{epochs}] "
              f"Loss={total_loss/n:.4f} "
              f"L_cls={total_cls/n:.4f} "
              f"L_rec={total_rec/n:.4f} "
              f"L_cov={total_cov/n:.4f}")

    # -------------------
    # Save models
    # -------------------
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "probe": probe.state_dict()
    }, save_path)
    print(f"✅ Pretrained models saved to {save_path}")


