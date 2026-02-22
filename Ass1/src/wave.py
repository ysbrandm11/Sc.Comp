import numpy as np

def phi_1(x):
    return np.sin(2.0 * np.pi * x)

def phi_2(x):
    return np.sin(5.0 * np.pi * x)

def phi_3(x):
    y = np.zeros_like(x, dtype=float)
    mask = (x > 1.0/5.0) & (x < 2.0/5.0)
    y[mask] = np.sin(5*np.pi*x[mask])
    return y
