import numpy as np
from Solvers import jacobi, gauss_seidel, sor

def make_c(Nx=50, Ny=50):
    """
    c has shape (Ny, Nx) with:
      - y=0 bottom row fixed to 0
      - y=1 top row fixed to 1
      - periodic in x (handled by modulo indexing in the solver)
      - initial condition: 0 for 0 <= y < 1 (interior starts at 0)
    """
    c = np.zeros((Ny, Nx), dtype=float)
    c[0, :]  = 0.0
    c[-1, :] = 1.0

    return c

c = make_c(Nx=50, Ny=50)

c = gauss_seidel(c, max_iterations=100, eps=1e-6, return_delta=False)
