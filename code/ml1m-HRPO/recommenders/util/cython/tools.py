import numpy as np


float_type = np.float32


def is_ndarray(instance, dtype=None):
    if not isinstance(instance, np.ndarray):
        return False
    if dtype is None:
        return True
    return instance.dtype == dtype
