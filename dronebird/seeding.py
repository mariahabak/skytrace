"""Fixing every source of randomness so that a run can be repeated.

Author: Mariah Habak
Reference: PyTorch "Reproducibility" notes,
https://pytorch.org/docs/stable/notes/randomness.html
"""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # the NumPy baseline does not need PyTorch
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
