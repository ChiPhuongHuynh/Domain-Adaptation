import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across PyTorch, NumPy, and Python's random module.
    Optionally sets PYTHONHASHSEED for consistent hashing.

    Args:
        seed (int): The desired seed value. Default is 42.
    """
    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)

    # Set seed for PyTorch on CUDA (GPU) if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups

    # Optional: Set PYTHONHASHSEED for consistent hashing behavior
    # This can affect dictionary and set order, which might impact reproducibility in some cases.
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Optional: Ensure deterministic behavior for some PyTorch operations
    # Note: This can sometimes lead to performance penalties.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Disable auto-tuning for CUDNN
