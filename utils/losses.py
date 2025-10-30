import torch
import torch.nn.functional as F

eps = 1e-6

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
