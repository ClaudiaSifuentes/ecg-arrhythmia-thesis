import os
import random

import numpy as np
import torch


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_determinism(seed: int, *, num_threads: int | None = None) -> None:
    """Seed every RNG and force deterministic kernels.

    - cuDNN: deterministic=True, benchmark=False (benchmark picks the fastest
      algorithm per shape at runtime, which can differ between runs).
    - torch.use_deterministic_algorithms(True): raises on ops without a
      deterministic implementation instead of silently varying.
    - CUBLAS_WORKSPACE_CONFIG must be set before the first cuBLAS call for
      deterministic matmuls on CUDA >= 10.2.
    - num_threads: CPU reductions are deterministic for a fixed thread count;
      fixing it removes one more source of run-to-run drift across machines.
    """

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_global_seeds(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive numpy/random seeds from torch's seed."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
