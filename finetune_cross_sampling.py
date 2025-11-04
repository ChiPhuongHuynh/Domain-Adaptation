import torch
from data.loader import get_dataloader
from models.models import SplitEncoder, SplitDecoder, LinearProbe
from utils.aligner import finetune_entropy
from experiments.compared_shared_encoder import compare_shared_encoder_alignment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 784
output_dim = 784
latent_dim = 64
signal_dim = 32
num_classes = 10

encoder = SplitEncoder(input_dim=input_dim, latent_dim=latent_dim, signal_dim=signal_dim).to(device)
decoder = SplitDecoder(latent_dim=latent_dim, output_dim=output_dim).to(device)
probe = LinearProbe()

# --- load your pretrained MNIST model ---
ckpt = torch.load("artifacts/mnist/mnist_pretrained_usage_swap_asym.pt", map_location=device)
encoder.load_state_dict(ckpt["encoder"])
decoder.load_state_dict(ckpt["decoder"])
probe.load_state_dict(ckpt["probe"])

loader_M = get_dataloader("mnist", batch_size=256, train=False)
loader_U = get_dataloader("usps", batch_size=256, train=False)

print("== Before finetune ==")
_ = compare_shared_encoder_alignment(encoder, decoder, loader_M, loader_U, device, visualize=False)

# --- short finetune ---
finetune_entropy(
    encoder, decoder, probe,
    loader_mnist=loader_M,
    loader_usps=loader_U,
    device=device,
    lambda_cls=1.0, lambda_rec=0.5, lambda_ent=0.3,
    epochs=3, lr=1e-4
)

print("== After finetune ==")
_ = compare_shared_encoder_alignment(encoder, decoder, loader_M, loader_U, device, visualize=False)
