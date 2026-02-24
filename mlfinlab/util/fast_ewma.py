"""
This module contains an implementation of an exponentially weighted moving average based on sample size.
The inspiration and context for this code was from a blog post by writen by Maksim Ivanov:
https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
"""

# Imports
import numpy as np
from numba import jit
from numba import float64
from numba import int64


@jit((float64[:], int64), nopython=False, nogil=True)
def ewma(arr_in, window):  # pragma: no cover
    """
    Exponentially weighted moving average specified by a decay ``window`` to provide better adjustments for
    small windows via:
        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    :param arr_in: (np.ndarray), (float64) A single dimensional numpy array
    :param window: (int64) The decay window, or 'span'
    :return: (np.ndarray) The EWMA vector, same length / shape as ``arr_in``
    """
    alpha = 2.0 / (window + 1.0)
    n = arr_in.shape[0]
    out = np.empty(n, dtype=np.float64)
    num = 0.0
    den = 0.0
    for i in range(n):
        num = arr_in[i] + (1.0 - alpha) * num
        den = 1.0 + (1.0 - alpha) * den
        out[i] = num / den
    return out
