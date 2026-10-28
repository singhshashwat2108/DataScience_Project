"""Seed every source of randomness used in training (Python, NumPy, torch, hash seed).

PyG's negative_sampling draws from Python's `random`, which was not seeded before, so two
runs with the same seed gave different embeddings. CPU only: no CUDA calls are made.
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)   # only affects child processes; start the run with it set
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():              # no-op on CPU-only machines
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
