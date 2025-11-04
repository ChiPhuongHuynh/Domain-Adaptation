import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

eps = 1e-6

# ============================================
# Label Adversarial Probe Loss for z_n
# ============================================

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    """Apply gradient reversal to x."""
    return GradReverse.apply(x, lambd)

def label_adversarial_probe_loss(probe_logits, true_labels):
    """
    Cross-entropy loss for the nuisance probe (predicting digit class from z_n).
    During encoder update, gradients are reversed so that the encoder removes label info.

    Args:
        probe_logits: output of probe network (B x num_classes)
        true_labels: ground-truth digit labels (0-9)
    Returns:
        loss tensor, metrics dict
    """
    ce = F.cross_entropy(probe_logits, true_labels)
    return ce, {"adv_ce": ce.item()}

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda' if torch.cuda.is_available() else 'cpu')

def ssim_reconstruction_loss(x_original, x_reconstructed, beta=0.1, gamma=0.5):
    """
    Combined MSE + L1 + SSIM reconstruction loss.
    Args:
        x_original: (B,C,H,W) or (B,H,W)
        x_reconstructed: same shape
    """
    # --- ensure correct shape ---
    if x_original.dim() == 2:  # (B,784)
        side = int(x_original.size(1) ** 0.5)
        x_original = x_original.view(-1, 1, side, side)
        x_reconstructed = x_reconstructed.view(-1, 1, side, side)
    elif x_original.dim() == 3:  # (B,H,W)
        x_original = x_original.unsqueeze(1)
        x_reconstructed = x_reconstructed.unsqueeze(1)

    # --- normalize and clamp ---
    x_original = torch.clamp(x_original, 0.0, 1.0)
    x_reconstructed = torch.clamp(x_reconstructed, 0.0, 1.0)

    # --- pixel-wise losses ---
    mse = F.mse_loss(x_reconstructed, x_original)
    l1  = F.l1_loss(x_reconstructed, x_original)

    # --- structural similarity ---
    ssim_val = ssim_metric(x_reconstructed, x_original)
    ssim_loss = 1.0 - ssim_val

    total = mse + beta * l1 + gamma * ssim_loss

    return total, {
        "mse": mse.item(),
        "l1": l1.item(),
        "ssim": ssim_val.item(),
        "total": total.item()
    }


def conservative_reconstruction_loss(x_original, x_reconstructed, beta=0.1):
    x_original = x_original.view(x_original.size(0),-1)
    x_reconstructed = x_reconstructed.view(x_reconstructed.size(0), -1)
    mse = F.mse_loss(x_reconstructed, x_original)
    l1 = F.l1_loss(x_reconstructed, x_original)

    return mse + beta*l1, {"mse":mse.item(), "l1":l1.item()}

def probe_loss(logits, labels):
    return F.cross_entropy(logits, labels)

def covariance_penalty(z_signal, z_nuis):
    z_signal = z_signal - z_signal.mean(0, keepdim=True)
    z_nuis = z_nuis - z_nuis.mean(0, keepdim=True)

    cov = torch.matmul(z_signal.T, z_nuis) / z_signal.size(0)
    return (cov**2).mean()

def get_weights(epoch, max_epochs):
    if epoch < max_epochs * 0.3:
        return 0.7, 0.3, 0.01
    elif epoch < max_epochs * 0.6:
        return 0.5, 0.4, 0.1
    else:
        return 0.3, 0.5, 0,2
    
def normalize_batch(z):
    # zero mean per dim and unit-std per dim (batch stats)
    zc = z - z.mean(dim=0, keepdim=True)
    std = zc.std(dim=0, keepdim=True)
    return zc / (std + eps)

def cross_covariance_norm(z_s, z_n):
    """
    Frobenius norm of cross-covariance between z_s and z_n (batch_normalized).
    Returns a scalar
    """
    zs = z_s - z_s.mean(dim=0, keepdim=True)
    zn = z_n - z_n.mean(dim=0, keepdim=True)
    cov = (zs.t() @ zn) / (zs.size(0) - 1.0)
    return torch.norm(cov)

def projection_penalty(z_s, z_n, ridge=1e-4):
    """
    Penalize projection of z_n onto span(z_s). 
    Solve linear least squares for A: z_n ~ z_s @ A,
    penalize ||z_s @ A||^2. Small ridge for better stability
    """
    zs = z_s - z_s.mean(dim=0, keepdim=True)
    zn = z_n - z_n.mean(dim=0, keepdim=True)

    G = zs.t() @ zs
    d_s = G.shape[0]
    G = G + ridge * torch.eye(d_s, device = G.device)

    A = torch.linalg.solve(G, zs.t() @ zn)

    zn_proj = zs @ A

    return (zn_proj.pow(2).sum(dim=1).mean())
