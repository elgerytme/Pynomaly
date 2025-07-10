import os
import random

import numpy as np


def fix_random_seeds():
    """Fix random seeds for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Auto-fix seeds when imported
fix_random_seeds()
