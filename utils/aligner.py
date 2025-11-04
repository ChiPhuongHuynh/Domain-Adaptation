# experiments/finetune_entropy.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.losses import ssim_reconstruction_loss, grad_reverse

def entropy_minimization(logits):
    """Entropy of softmax outputs: lower is more confident."""
    p = torch.softmax(logits, dim=1)
    return -(p * torch.log(p.clamp_min(1e-6))).sum(dim=1).mean()

def _entropy_min(logits):
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
            xhatU = decoder(torch.cat([z_sU.detach(), z_nU], dim=1))

            # --- reshape flat decoder outputs back into 2D images ---
            side = int(xM.size(-1))  # 28 for MNIST
            xhatM = xhatM.view(-1, 1, side, side)
            xhatU = xhatU.view(-1, 1, side, side)
            L_recM, _ = ssim_reconstruction_loss(xM, xhatM)
            L_recU, _ = ssim_reconstruction_loss(xU, xhatU)
            L_rec = L_recM + L_recU

            # ---- USPS entropy loss (no labels) ----
            logitsU = probe(z_sU.detach())
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


def finetune_entropy_detach_stabilized(
    encoder, decoder, probe,
    loader_mnist, loader_usps,
    device,
    lambda_cls=1.0, lambda_rec=0.5, lambda_ent=0.3,
    lambda_zvar=2e-3, lambda_swap=1e-2, lambda_mom = 0.005,
    epochs=3, lr=1e-4
):
    """
    USPS finetune with detached entropy (safe version) + two stabilizers:
      • z_n variance clamp (keeps nuisance variance bounded)
      • Δswap regularizer (discourages decoder over-reliance on z_n)
    """

    encoder.train()
    decoder.train()
    probe.train()

    opt = torch.optim.AdamW(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(probe.parameters()),
        lr=lr, weight_decay=1e-4
    )

    itM, itU = iter(loader_mnist), iter(loader_usps)

    for ep in range(epochs):
        total_loss = total_cls = total_rec = total_ent = 0.0
        total_zvar = total_swap = 0.0

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
            xhatU = decoder(torch.cat([z_sU.detach(), z_nU], dim=1))

            side = int(xM.size(-1))
            xhatM = xhatM.view(-1, 1, side, side)
            xhatU = xhatU.view(-1, 1, side, side)

            L_recM, _ = ssim_reconstruction_loss(xM, xhatM)
            L_recU, _ = ssim_reconstruction_loss(xU, xhatU)
            L_rec = L_recM + L_recU

            # ---- USPS entropy loss (no labels, detached signal) ----
            logitsU = probe(z_sU.detach())
            L_ent = entropy_minimization(logitsU)

            # ---- z_n variance clamp ----
            with torch.no_grad():
                v_ref = z_nM.var(dim=0, unbiased=False).detach()
            vM = z_nM.var(dim=0, unbiased=False)
            vU = z_nU.var(dim=0, unbiased=False)
            L_zvar = (vM - v_ref).abs().mean() + (vU - v_ref).abs().mean()

            # ---- Δswap regularizer ----
            z_nM_mean = z_nM.mean(dim=0, keepdim=True)
            z_nU_mean = z_nU.mean(dim=0, keepdim=True)
            xhatM_swap = decoder(torch.cat([z_sM, z_nM_mean.expand_as(z_nM)], dim=1))
            xhatU_swap = decoder(torch.cat([z_sU, z_nU_mean.expand_as(z_nU)], dim=1))
            xhatM_swap = xhatM_swap.view(-1, 1, side, side)
            xhatU_swap = xhatU_swap.view(-1, 1, side, side)
            L_swap = (xM - xhatM_swap).abs().mean() + (xU - xhatU_swap).abs().mean()

            # moment matching term for signal
            mu_M, mu_U = z_sM.mean(dim=0), z_sU.mean(dim=0)
            std_M, std_U = z_sM.std(dim=0), z_sU.std(dim=0)
            L_mom = (mu_M - mu_U).pow(2).mean() + (std_M - std_U).pow(2).mean()

            # ---- Combine ----
            loss = (
                lambda_cls * L_cls +
                lambda_rec * L_rec +
                lambda_ent * L_ent +
                lambda_zvar * L_zvar +
                lambda_swap * L_swap +
                lambda_mom * L_mom
            )

            # ---- Update ----
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 3.0)
            opt.step()

            # ---- Track ----
            bsz = xM.size(0)
            total_loss += loss.item() * bsz
            total_cls  += L_cls.item() * bsz
            total_rec  += L_rec.item() * bsz
            total_ent  += L_ent.item() * bsz
            total_zvar += L_zvar.item() * bsz
            total_swap += L_swap.item() * bsz

        denom = min(len(loader_mnist), len(loader_usps)) * xM.size(0)
        print(f"[Finetune Detach+Stabilized {ep+1}/{epochs}] "
              f"Loss={total_loss/denom:.4f} "
              f"Cls={total_cls/denom:.4f} "
              f"Rec={total_rec/denom:.4f} "
              f"Ent={total_ent/denom:.4f} "
              f"Zvar={total_zvar/denom:.4f} "
              f"Swap={total_swap/denom:.4f}")


def finetune_domain_adversary(
    encoder,
    domain_probe,
    loader_mnist,
    loader_usps,
    device,
    lambda_adv=0.2,
    lr=1e-4,
    epochs=3,
    save_path="artifacts/mnist/mnist_domain_adversary.pt",
):
    """
    Finetune the encoder with a domain adversary to make z_s domain-invariant.

    Args:
        encoder: trained encoder (produces z_s, z_n)
        domain_probe: small MLP classifier with 2 outputs (MNIST=0, USPS=1)
        loader_mnist, loader_usps: DataLoaders for each domain
        lambda_adv: strength of gradient reversal
        lr: learning rate (small; 1e-4–3e-4 works well)
        epochs: number of adversarial finetuning epochs
        save_path: output checkpoint
    """

    encoder.train()
    domain_probe.train()

    # freeze nuisance-related layers if needed (optional)
    for p in encoder.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW([
    {"params": encoder.parameters(), "lr": 1e-4},
    {"params": domain_probe.parameters(), "lr": 5e-5}  # slower probe
    ])


    itM, itU = iter(loader_mnist), iter(loader_usps)

    for ep in range(epochs):
        total_loss = total_acc = 0.0

        # ensure equal domain samples per epoch
        for _ in tqdm(range(min(len(loader_mnist), len(loader_usps))),
                      desc=f"[DomainAdv {ep+1}/{epochs}]"):

            # ---- Sample one MNIST and one USPS batch ----
            try:
                xM, _ = next(itM)
            except StopIteration:
                itM = iter(loader_mnist)
                xM, _ = next(itM)

            try:
                xU, _ = next(itU)
            except StopIteration:
                itU = iter(loader_usps)
                xU, _ = next(itU)

            xM, xU = xM.to(device), xU.to(device)

            # ---- Domain labels: MNIST=0, USPS=1 ----
            y_dom_M = torch.zeros(xM.size(0), dtype=torch.long, device=device)
            y_dom_U = torch.ones(xU.size(0), dtype=torch.long, device=device)

            # combine into a single batch
            x = torch.cat([xM, xU], dim=0)
            y_dom = torch.cat([y_dom_M, y_dom_U], dim=0)

            # ---- Encode (we only use z_s here) ----
            z_s, _ = encoder(x)

            # ---- Gradient reversal ----
            z_s_rev = grad_reverse(z_s, lambd=lambda_adv)
            logits = domain_probe(z_s_rev)

            # ---- Domain classification loss ----
            L_dom = F.cross_entropy(logits, y_dom)

            # ---- Backprop ----
            opt.zero_grad(set_to_none=True)
            L_dom.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            opt.step()

            # ---- Track accuracy ----
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == y_dom).float().mean().item()

            total_loss += L_dom.item() * x.size(0)
            total_acc  += acc * x.size(0)

        n = 2 * min(len(loader_mnist), len(loader_usps)) * xM.size(0)
        print(f"[DomainAdv Epoch {ep+1}/{epochs}] "
              f"L_dom={total_loss/n:.4f}  DomAcc={total_acc/n:.4f}")

    # -------------------
    # Save updated encoder + domain probe
    # -------------------
    torch.save({
        "encoder": encoder.state_dict(),
        "domain_probe": domain_probe.state_dict(),
    }, save_path)
    print(f"✅ Domain-adversarial finetune saved to {save_path}")

def finetune_entropy_detach_usps_contrastive(
    encoder, decoder, probe,
    loader_mnist, loader_usps,
    device,
    domain_probe=None,         # optional domain classifier for z_n
    lambda_cls=1.0, lambda_rec=0.5, lambda_ent=0.3, lambda_dom=0.2,
    epochs=3, lr=1e-4,
    save_path="artifacts/mnist/mnist_finetuned_usps_detach_contrastive.pt"
):
    """
    Finetune with USPS detachment + optional domain-contrastive loss on nuisance.

    Behavior:
    - MNIST: supervised classification + reconstruction (normal grads)
    - USPS: detached z_s, reconstruction and entropy-only adaptation
    - Optional: domain_probe encourages z_n to encode domain information explicitly
    """

    encoder.train()
    decoder.train()
    probe.train()
    if domain_probe is not None:
        domain_probe.train()

    # optimizer includes domain_probe if provided
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(probe.parameters())
    if domain_probe is not None:
        params += list(domain_probe.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    itM, itU = iter(loader_mnist), iter(loader_usps)

    for ep in range(epochs):
        total_loss, total_cls, total_rec, total_ent, total_dom = 0, 0, 0, 0, 0

        for _ in range(min(len(loader_mnist), len(loader_usps))):
            # ---- get batches ----
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

            # ---- encode ----
            z_sM, z_nM = encoder(xM)
            z_sU, z_nU = encoder(xU)

            # ---- MNIST classification ----
            logitsM = probe(z_sM)
            L_cls = F.cross_entropy(logitsM, yM)

            # ---- reconstruction ----
            xhatM = decoder(torch.cat([z_sM, z_nM], dim=1))
            xhatU = decoder(torch.cat([z_sU.detach(), z_nU], dim=1))  # USPS z_s detached

            side = int(xM.size(-1))  # e.g. 28 for MNIST
            xhatM = xhatM.view(-1, 1, side, side)
            xhatU = xhatU.view(-1, 1, side, side)
            L_recM, _ = ssim_reconstruction_loss(xM, xhatM)
            L_recU, _ = ssim_reconstruction_loss(xU, xhatU)
            L_rec = L_recM + L_recU

            # ---- USPS entropy loss (no labels) ----
            logitsU = probe(z_sU.detach())   # USPS z_s detached from probe grads too
            L_ent = entropy_minimization(logitsU)

            # ---- domain-contrastive loss on z_n (optional) ----
            if domain_probe is not None:
                y_domM = torch.zeros(z_nM.size(0), dtype=torch.long, device=device)
                y_domU = torch.ones(z_nU.size(0), dtype=torch.long, device=device)
                z_n_concat = torch.cat([z_nM, z_nU], dim=0)
                y_dom_concat = torch.cat([y_domM, y_domU], dim=0)

                logits_dom = domain_probe(z_n_concat)
                L_dom = F.cross_entropy(logits_dom, y_dom_concat)
            else:
                L_dom = torch.tensor(0.0, device=device)

            # ---- combine losses ----
            loss = (
                lambda_cls * L_cls +
                lambda_rec * L_rec +
                lambda_ent * L_ent +
                lambda_dom * L_dom
            )

            # ---- update ----
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            opt.step()

            # ---- track ----
            bsz = xM.size(0)
            total_loss += loss.item() * bsz
            total_cls  += L_cls.item() * bsz
            total_rec  += L_rec.item() * bsz
            total_ent  += L_ent.item() * bsz
            total_dom  += L_dom.item() * bsz

        denom = min(len(loader_mnist), len(loader_usps)) * xM.size(0)
        print(f"[Finetune Detach+Contrastive {ep+1}/{epochs}] "
              f"Loss={total_loss/denom:.4f} "
              f"Cls={total_cls/denom:.4f} "
              f"Rec={total_rec/denom:.4f} "
              f"Ent={total_ent/denom:.4f} "
              f"Dom={total_dom/denom:.4f}")

    # ---- save ----
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "probe": probe.state_dict(),
        "domain_probe": None if domain_probe is None else domain_probe.state_dict()
    }, save_path)
    print(f"✅ USPS finetune (detach + contrastive) saved to {save_path}")

