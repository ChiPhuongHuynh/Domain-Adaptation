# experiments/finetune_entropy.py
import torch
import torch.nn.functional as F
from utils.losses import ssim_reconstruction_loss

def entropy_minimization(logits):
    """Entropy of softmax outputs: lower is more confident."""
    p = torch.softmax(logits, dim=1)
    return -(p * torch.log(p.clamp_min(1e-6))).sum(dim=1).mean()


def finetune_entropy(
    encoder, decoder, probe,
    loader_mnist, loader_usps,
    device,
    lambda_cls=1.0, lambda_rec=0.5, lambda_ent=0.3,
    epochs=3, lr=1e-4
):
    """
    Unsupervised domain adaptation finetune:
    - Keep MNIST supervision (labels)
    - Add unlabeled USPS batches (entropy minimization)
    - Small learning rate, few epochs
    """

    encoder.train()
    decoder.train()
    probe.train()

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()) + list(probe.parameters()),
        lr=lr, weight_decay=1e-4
    )

    itM, itU = iter(loader_mnist), iter(loader_usps)

    for ep in range(epochs):
        total_loss, total_cls, total_rec, total_ent = 0, 0, 0, 0

        for _ in range(min(len(loader_mnist), len(loader_usps))):
            # ---- Get batches ----
            try:
                xM, yM = next(itM)
            except StopIteration:
                itM = iter(loader_mnist)
                xM, yM = next(itM)
            try:
                xU, _ = next(itU)
            except StopIteration:
                itU = iter(loader_usps)
                xU, _ = next(itU)

            xM, yM, xU = xM.to(device), yM.to(device), xU.to(device)

            # ---- Encode ----
            z_sM, z_nM = encoder(xM)
            z_sU, z_nU = encoder(xU)

            # ---- MNIST classification ----
            logitsM = probe(z_sM)
            L_cls = F.cross_entropy(logitsM, yM)

            # ---- Reconstruction ----
            xhatM = decoder(torch.cat([z_sM, z_nM], dim=1))
            xhatU = decoder(torch.cat([z_sU, z_nU], dim=1))

            # --- reshape flat decoder outputs back into 2D images ---
            side = int(xM.size(-1))  # 28 for MNIST
            xhatM = xhatM.view(-1, 1, side, side)
            xhatU = xhatU.view(-1, 1, side, side)
            L_recM, _ = ssim_reconstruction_loss(xM, xhatM)
            L_recU, _ = ssim_reconstruction_loss(xU, xhatU)
            L_rec = L_recM + L_recU

            # ---- USPS entropy loss (no labels) ----
            logitsU = probe(z_sU)
            L_ent = entropy_minimization(logitsU)

            # ---- Combine ----
            loss = (
                lambda_cls * L_cls +
                lambda_rec * L_rec +
                lambda_ent * L_ent
            )

            # ---- Update ----
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            opt.step()

            # ---- Track ----
            bsz = xM.size(0)
            total_loss += loss.item() * bsz
            total_cls  += L_cls.item() * bsz
            total_rec  += L_rec.item() * bsz
            total_ent  += L_ent.item() * bsz

        denom = min(len(loader_mnist), len(loader_usps)) * xM.size(0)
        print(f"[Finetune {ep+1}/{epochs}] "
              f"Loss={total_loss/denom:.4f} "
              f"Cls={total_cls/denom:.4f} "
              f"Rec={total_rec/denom:.4f} "
              f"Ent={total_ent/denom:.4f}")
