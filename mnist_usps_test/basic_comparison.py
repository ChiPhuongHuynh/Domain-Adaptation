from experiments.compare_shared_nuisance import compare_shared_nuisance_alignment
from data.loader import get_dataloader
from models.models import SplitEncoder
from models.models import SplitDecoder
import torch

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

results = compare_shared_nuisance_alignment(
    encoder_M, decoder_M,
    encoder_U, decoder_U,
    loader_M, loader_U,
    device=device,
    n_samples=1000,
    visualize=False
)

print(results)
