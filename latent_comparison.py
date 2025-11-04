import torch
from data.loader import get_dataloader
from models.models import SplitEncoder
from models.models import SplitDecoder
from metrics.mmd import mmd_gaussian


device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained models
ckpt_M = torch.load("artifacts/mnist/mnist_pretrained.pt", map_location=device)
ckpt_U = torch.load("artifacts/usps/usps_pretrained.pt", map_location=device)

input_dim = 784
output_dim = 784
latent_dim = 64
signal_dim = 32
num_classes = 10

encoder_M = SplitEncoder(input_dim=input_dim, latent_dim=latent_dim, signal_dim=signal_dim).to(device)
decoder_M = SplitDecoder(latent_dim=latent_dim, output_dim=output_dim).to(device)

encoder_U = SplitEncoder(input_dim=input_dim, latent_dim=latent_dim, signal_dim=signal_dim).to(device)
decoder_U = SplitDecoder(latent_dim=latent_dim, output_dim=output_dim).to(device)

encoder_M.load_state_dict(ckpt_M["encoder"])
decoder_M.load_state_dict(ckpt_M["decoder"])
encoder_U.load_state_dict(ckpt_U["encoder"])
decoder_U.load_state_dict(ckpt_U["decoder"])

loader_M = get_dataloader("mnist", batch_size=256, train=False)
loader_U = get_dataloader("usps", batch_size=256, train=False)

def get_latents_and_recons(encoder, decoder, loader, n_batches=10):
    xs, zs, zn, xh, xh_s = [], [], [], [], []
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device)
        z_s, z_n = encoder(x)
        x_hat = decoder(torch.cat([z_s, z_n], dim=1))
        x_hat_s = decoder(torch.cat([z_s, torch.zeros_like(z_n)], dim=1))
        xs.append(x.view(len(x), -1))
        zs.append(z_s)
        zn.append(z_n)
        xh.append(x_hat.view(len(x), -1))
        xh_s.append(x_hat_s.view(len(x), -1))
    return map(lambda t: torch.cat(t, dim=0).cpu(), (xs, zs, zn, xh, xh_s))

# extract
xS, zsS, znS, xhS, xhsS = get_latents_and_recons(encoder_M, decoder_M, loader_M)
xT, zsT, znT, xhT, xhsT = get_latents_and_recons(encoder_U, decoder_U, loader_U)

# compute MMD
mmds = {
    "original": mmd_gaussian(xS, xT),
    "signal_latent": mmd_gaussian(zsS, zsT),
    "nuisance_latent": mmd_gaussian(znS, znT),
    "reconstruction": mmd_gaussian(xhS, xhT),
    "signal_recon": mmd_gaussian(xhsS, xhsT),
}

print("\n=== MMD Results ===")
for k, v in mmds.items():
    print(f"{k:20s}: {v:.4f}")
