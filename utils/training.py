import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.losses import (
    probe_loss,
    conservative_reconstruction_loss,
    cross_covariance_norm,
    covariance_penalty,
    projection_penalty,
    normalize_batch,
    ssim_reconstruction_loss,
    label_adversarial_probe_loss
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
            L_rec, rec_stats = ssim_reconstruction_loss(
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

def pretrain_round_robin(
    encoder, decoder, probe, label_probe, dataloader, device,
    lambda_cls=1.0, lambda_rec=0.5, lambda_preserve=1.0,
    lambda_proj=0.1, lambda_cycle_nuisance=1.0,
    lambda_adv=0.3,                 # strength of “remove labels from z_n”
    probe_steps=1, enc_steps=1,     # turn-based ratio
    epochs=20, save_path="mnist_pretrained_alt.pt",
    lr_enc=1e-3, lr_probe=1e-3
):
    encoder.train(); decoder.train(); probe.train(); label_probe.train()

    opt_enc = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(probe.parameters()),
        lr=lr_enc
    )
    opt_probe = torch.optim.Adam(label_probe.parameters(), lr=lr_probe)

    for epoch in range(epochs):
        running = {"loss":0,"cls":0,"rec":0,"adv_p":0,"adv_e":0,"pacc":0,"cacc":0}
        n_seen = 0

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            # ----------------------------
            # PHASE A: train label_probe
            # ----------------------------
            for _ in range(probe_steps):
                encoder.eval(); label_probe.train()
                with torch.no_grad():
                    z_s, z_n = encoder(x)
                logits_p = label_probe(z_n.detach())
                L_adv_probe, _ = label_adversarial_probe_loss(logits_p, y)

                opt_probe.zero_grad(set_to_none=True)
                L_adv_probe.backward()
                opt_probe.step()

            # ----------------------------
            # PHASE B: train encoder/cls/decoder
            # ----------------------------
            for _ in range(enc_steps):
                encoder.train(); label_probe.eval()
                z_s, z_n = encoder(x)

                # (1) classifier on z_s
                logits = probe(z_s)
                L_cls = F.cross_entropy(logits, y)

                # (2) reconstruction (SSIM + MSE/L1)
                x_hat = decoder(torch.cat([z_s.detach(), z_n], dim=1))
                L_rec, _ = ssim_reconstruction_loss(x.view(x.size(0), -1), x_hat)

                # (3) preserve signal + nuisance cycle
                z_s_p, z_n_p = encoder(x_hat)
                L_preserve = torch.norm(z_s - z_s_p, dim=1).mean()
                L_cycle_nuis = torch.norm(z_n_p - z_n, dim=1).mean()

                # (4) optional projection (small)
                L_proj = projection_penalty(z_s, z_n)

                # (5) adversarial term: make probe fail on z_n (maximize CE)
                logits_p_frozen = label_probe(z_n)  # probe weights frozen
                L_adv_enc, _ = label_adversarial_probe_loss(logits_p_frozen, y)

                loss_enc = (
                    lambda_cls * L_cls
                    + lambda_rec * L_rec
                    + lambda_preserve * L_preserve
                    + lambda_cycle_nuisance * L_cycle_nuis
                    + lambda_proj * L_proj
                    - lambda_adv * L_adv_enc            # maximize probe CE on z_n
                )

                opt_enc.zero_grad(set_to_none=True)
                loss_enc.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
                opt_enc.step()

            # ------------- Logging -------------
            with torch.no_grad():
                # probe accuracy (on current z_n)
                p_pred = logits_p.argmax(dim=1)
                p_acc = (p_pred == y).float().mean().item()
                # classifier accuracy (on z_s)
                c_pred = logits.argmax(dim=1)
                c_acc = (c_pred == y).float().mean().item()

            bsz = x.size(0); n_seen += bsz
            running["loss"] += loss_enc.item() * bsz
            running["cls"]  += L_cls.item() * bsz
            running["rec"]  += L_rec.item() * bsz
            running["adv_p"]+= L_adv_probe.item() * bsz
            running["adv_e"]+= L_adv_enc.item() * bsz
            running["pacc"] += p_acc * bsz
            running["cacc"] += c_acc * bsz

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Loss={running['loss']/n_seen:.3f} "
              f"L_cls={running['cls']/n_seen:.3f} "
              f"L_rec={running['rec']/n_seen:.3f} "
              f"L_advProbe(min)={running['adv_p']/n_seen:.3f} "
              f"L_advEnc(max)={running['adv_e']/n_seen:.3f} "
              f"ProbeAcc(z_n)={running['pacc']/n_seen:.2f} "
              f"ClsAcc(z_s)={running['cacc']/n_seen:.2f}")

    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "probe": probe.state_dict(),
        "label_probe": label_probe.state_dict(),
    }, save_path)
    print(f"✅ Saved to {save_path}")


def pretrain_usage_swap(
    encoder, decoder, probe, dataloader, device,
    lambda_cls=1.0, lambda_rec=0.5, lambda_preserve=1.0,
    lambda_cov=0.5, lambda_proj=0.1, lambda_cov_cycle=0.5,
    lambda_cycle_nuisance=1.0, lambda_use=0.05,
    usage_margin=0.02,
    epochs=20, save_path="artifacts/mnist/mnist_pretrained_usage_swap.pt"
):
    """
    Pretraining variant that adds a nuisance-usage penalty (swap loss).

    The swap loss encourages the decoder to depend more on z_n:
        L_use = max(0, margin - mean(|x - x_swapped|))
    where x_swapped is decoded with shuffled z_n.

    All other losses are identical to the base pretrain() function.
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
        total_loss = total_cls = total_rec = total_cov = total_use = 0.0

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            # -------------------
            # Forward pass
            # -------------------
            z_s, z_n = encoder(x)
            logits = probe(z_s)

            # -------------------
            # (1) Classification
            # -------------------
            L_cls = F.cross_entropy(logits, y)

            # -------------------
            # (2) Reconstruction
            # -------------------
            z_n_noisy = z_n + torch.randn_like(z_n) * 0.01
            x_hat = decoder(torch.cat([z_s.detach(), z_n_noisy], dim=1))
            L_rec, rec_stats = ssim_reconstruction_loss(
                x.view(x.size(0), -1), x_hat
            )

            # -------------------
            # (3) Signal preservation & cycle consistency
            # -------------------
            z_s_p, z_n_p = encoder(x_hat)
            L_preserve = torch.norm(z_s - z_s_p, dim=1).mean()
            L_cycle_nuisance = torch.norm(z_n_p - z_n, dim=1).mean()

            # -------------------
            # (4) Cross-covariance and projection penalties
            # -------------------
            z_s_norm = normalize_batch(z_s)
            z_n_norm = normalize_batch(z_n)

            L_cov = cross_covariance_norm(z_s_norm, z_n_norm)
            L_cov_cycle = covariance_penalty(z_s_p, z_n_p)
            L_proj = projection_penalty(z_s, z_n)

            # -------------------
            # (5) Nuisance-usage (swap) loss
            # -------------------
            with torch.no_grad():
                perm = torch.randperm(z_n.size(0), device=device)
                z_n_shuf = z_n[perm]

            x_swap = decoder(torch.cat([z_s.detach(), z_n_shuf], dim=1))
            L_swap = (x_hat - x_swap).abs().mean()
            L_use = torch.clamp(usage_margin - L_swap, min=0.0)

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
                + lambda_use * L_use
            )

            # -------------------
            # Backpropagation
            # -------------------
            opt.zero_grad()
            loss.backward()
            opt.step()

            # -------------------
            # Logging accumulators
            # -------------------
            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_cls  += L_cls.item() * bsz
            total_rec  += L_rec.item() * bsz
            total_cov  += L_cov.item() * bsz
            total_use  += L_use.item() * bsz

        # -------------------
        # Epoch summary
        # -------------------
        n = len(dataloader.dataset)
        print(f"[Pretrain+Swap Epoch {epoch+1}/{epochs}] "
              f"Loss={total_loss/n:.4f} "
              f"L_cls={total_cls/n:.4f} "
              f"L_rec={total_rec/n:.4f} "
              f"L_cov={total_cov/n:.4f} "
              f"L_use={total_use/n:.4f}")

    # -------------------
    # Save models
    # -------------------
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "probe": probe.state_dict(),
    }, save_path)
    print(f"✅ Pretrained models saved to {save_path}")


def pretrain_usage_swap_balanced(
    encoder, decoder, probe, dataloader, device,
    lambda_cls=1.0, lambda_rec=0.5, lambda_preserve=1.0,
    lambda_cov=0.02, lambda_proj=0.1, lambda_cov_cycle=0.5,
    lambda_cycle_nuisance=1.0, lambda_use=0.05,
    usage_margin=0.02,
    epochs=20, save_path="artifacts/mnist/mnist_pretrained_usage_swap_balanced.pt"
):
    """
    Pretraining variant combining:
      • Supervised classification (z_s)
      • SSIM reconstruction
      • Latent preservation & consistency
      • Small decorrelation penalty (z_s ⟂ z_n)
      • Nuisance-usage (swap) penalty encouraging decoder to depend on z_n

    The decorrelation (λ_cov=0.1) restores separation stability after enabling z_n usage.
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
        total_loss = total_cls = total_rec = total_cov = total_use = 0.0

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            # -------------------
            # Forward pass
            # -------------------
            z_s, z_n = encoder(x)
            logits = probe(z_s)

            # (1) Classification
            L_cls = F.cross_entropy(logits, y)

            # (2) Reconstruction
            z_n_noisy = z_n + torch.randn_like(z_n) * 0.01
            x_hat = decoder(torch.cat([z_s.detach(), z_n_noisy], dim=1))
            L_rec, rec_stats = ssim_reconstruction_loss(
                x.view(x.size(0), -1), x_hat
            )

            # (3) Latent preservation & cycle
            z_s_p, z_n_p = encoder(x_hat)
            L_preserve = torch.norm(z_s - z_s_p, dim=1).mean()
            L_cycle_nuisance = torch.norm(z_n_p - z_n, dim=1).mean()

            # (4) Light decorrelation & projection
            z_s_norm = normalize_batch(z_s)
            z_n_norm = normalize_batch(z_n)
            L_cov_raw = cross_covariance_norm(z_s_norm, z_n_norm)
            L_cov = torch.tanh(L_cov_raw)
            L_cov_cycle = covariance_penalty(z_s_p, z_n_p)
            L_proj = projection_penalty(z_s, z_n)

            # (5) Nuisance-usage (swap) loss
            with torch.no_grad():
                perm = torch.randperm(z_n.size(0), device=device)
                z_n_shuf = z_n[perm]
            x_swap = decoder(torch.cat([z_s.detach(), z_n_shuf], dim=1))
            L_swap = (x_hat - x_swap).abs().mean()
            L_use = torch.clamp(usage_margin - L_swap, min=0.0)

            # (6) Total loss
            loss = (
                lambda_cls * L_cls
                + lambda_rec * L_rec
                + lambda_preserve * L_preserve
                + lambda_cov * L_cov        # <── RE-INTRODUCED, gentle weight
                + lambda_proj * L_proj
                + lambda_cov_cycle * L_cov_cycle
                + lambda_cycle_nuisance * L_cycle_nuisance
                + lambda_use * L_use
            )

            # Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Logging accumulators
            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_cls  += L_cls.item() * bsz
            total_rec  += L_rec.item() * bsz
            total_cov  += L_cov.item() * bsz
            total_use  += L_use.item() * bsz

        # Epoch summary
        n = len(dataloader.dataset)
        print(f"[Pretrain+Swap+Cov Epoch {epoch+1}/{epochs}] "
              f"Loss={total_loss/n:.4f} "
              f"L_cls={total_cls/n:.4f} "
              f"L_rec={total_rec/n:.4f} "
              f"L_cov={total_cov/n:.4f} "
              f"L_use={total_use/n:.4f}")

    # Save
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "probe": probe.state_dict(),
    }, save_path)
    print(f"✅ Pretrained models saved to {save_path}")

def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def pretrain_usage_swap_asym(
    encoder, decoder, probe, dataloader, device,
    lambda_cls=1.0,          # keep this dominant
    lambda_rec=0.5,          # recon guard
    lambda_use=0.05,         # swap usage (small)
    lambda_cov=0.02,         # very small, *asymmetric* decorrelation
    lambda_preserve=1.0,     # latent preservation
    lambda_cycle_nuisance=1.0,
    lambda_proj=0.1,
    usage_margin=0.02,
    warmup_epochs=2,         # brief classifier warmup
    epochs=20,
    lr=1e-3,
    save_path="mnist_pretrained_usage_swap_asym.pt",
):
    """
    - Warmup: train encoder+probe on L_cls only (stabilize z_s).
    - Main: add SSIM recon, swap-usage loss (flows to z_n+decoder only),
            and *asymmetric* decorrelation: cov( stopgrad(z_s), z_n ).
      This prevents the cov term from fighting the classifier over z_s.
    """

    encoder.train(); decoder.train(); probe.train()

    opt = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(probe.parameters()),
        lr=lr
    )

    # -----------------------
    # (A) Warmup classifier
    # -----------------------
    for ep in range(warmup_epochs):
        total_loss = total_cls = 0.0
        for x, y in tqdm(dataloader, desc=f"Warmup {ep+1}/{warmup_epochs}"):
            x, y = x.to(device), y.to(device)
            z_s, z_n = encoder(x)
            logits = probe(z_s)
            L_cls = F.cross_entropy(logits, y)
            loss = lambda_cls * L_cls

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            b = x.size(0)
            total_loss += loss.item() * b
            total_cls  += L_cls.item() * b
        n = len(dataloader.dataset)
        print(f"[Warmup {ep+1}/{warmup_epochs}] Loss={total_loss/n:.4f} L_cls={total_cls/n:.4f}")

    # --------------------------------
    # (B) Main training with asym cov
    # --------------------------------
    for epoch in range(epochs):
        total_loss = total_cls = total_rec = total_use = total_cov = 0.0

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            # Encode
            z_s, z_n = encoder(x)

            # (1) Classification on z_s (drives encoder z_s + probe)
            logits = probe(z_s)
            L_cls = F.cross_entropy(logits, y)

            # (2) Reconstruction (use z_s.detach() so recon path doesn’t fight classifier)
            z_n_noisy = z_n + 0.01 * torch.randn_like(z_n)
            x_hat = decoder(torch.cat([z_s.detach(), z_n_noisy], dim=1))
            L_rec, _ = ssim_reconstruction_loss(x.view(x.size(0), -1), x_hat)  # accepts (B,1,H,W) vs flat internally

            # Re-encode recon to keep latents stable
            z_s_p, z_n_p = encoder(x_hat)
            L_preserve = torch.norm(z_s - z_s_p, dim=1).mean()
            L_cycle_nuisance = torch.norm(z_n_p - z_n, dim=1).mean()

            # (3) Asymmetric decorrelation:
            #     Only update z_n to reduce correlation with the *fixed* z_s.
            with torch.no_grad():
                z_s_ng = z_s.clone()
            z_sn = normalize_batch(z_s_ng)         # stop-grad z_s
            z_nn = normalize_batch(z_n)            # gradients flow into z_n
            L_cov_asym = cross_covariance_norm(z_sn, z_nn)

            # (4) Projection penalty (light)
            L_proj = projection_penalty(z_s.detach(), z_n)  # also asym: don’t disturb z_s

            # (5) Nuisance usage (swap) — only z_n+decoder should bear this
            with torch.no_grad():
                perm = torch.randperm(z_n.size(0), device=device)
                z_n_shuf = z_n[perm]
            x_swap = decoder(torch.cat([z_s.detach(), z_n_shuf], dim=1))
            L_swap = (x_hat - x_swap).abs().mean()
            L_use = torch.clamp(usage_margin - L_swap, min=0.0)

            # Total loss (z_s is primarily steered by L_cls; z_n/decoder by L_rec/L_use/L_cov)
            loss = (
                lambda_cls * L_cls
                + lambda_rec * L_rec
                + lambda_preserve * L_preserve
                + lambda_cycle_nuisance * L_cycle_nuisance
                + lambda_proj * L_proj
                + lambda_use * L_use
                + lambda_cov * L_cov_asym
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            opt.step()

            b = x.size(0)
            total_loss += loss.item() * b
            total_cls  += L_cls.item() * b
            total_rec  += L_rec.item() * b
            total_use  += L_use.item() * b
            total_cov  += L_cov_asym.item() * b

        n = len(dataloader.dataset)
        print(f"[Pretrain+Swap(Asym) {epoch+1}/{epochs}] "
              f"Loss={total_loss/n:.4f} "
              f"L_cls={total_cls/n:.4f} "
              f"L_rec={total_rec/n:.4f} "
              f"L_use={total_use/n:.4f} "
              f"L_cov(asym)={total_cov/n:.4f}")

    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "probe": probe.state_dict(),
    }, save_path)
    print(f"✅ Pretrained models saved to {save_path}")
