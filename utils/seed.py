"""Global RNG seeding for reproducible training."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: some gsplat/cuda ops may remain nondeterministic.
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seed (pairs with ``torch.Generator.manual_seed``)."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    worker_seed = int(info.seed % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed: int, *, train: bool = True) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed) + (0 if train else 1))
    return g
