import numpy as np
from scipy.special import erfc


def analytical_c(y: np.ndarray, t: float, D: float = 1.0, n_terms: int = 200) -> np.ndarray:
    if t == 0:
        c = np.zeros_like(y)
        c[-1] = 1.0
        return c
    
    c = np.zeros_like(y, dtype=float)

    for i in range(n_terms):
        term1 = erfc((1 - y + 2 * i) /  (2 * np.sqrt(D * t)))
        term2 = erfc((1 + y + 2 * i) /  (2 * np.sqrt(D * t)))
        c += term1 - term2

    return c

def analytical_steady(y: np.ndarray) -> np.ndarray:
    return y