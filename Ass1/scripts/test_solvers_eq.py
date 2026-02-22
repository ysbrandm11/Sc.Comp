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

c_J, d_J= jacobi(c.copy(), max_iterations=5000, eps=1e-5, return_delta=True)
c_gs, d_gs = gauss_seidel(c.copy(), max_iterations=5000, eps=1e-5, return_delta=True)
c_sor, d_sor = sor(c.copy(), omega=1.85, max_iterations=5000, eps=1e-5, return_delta=True)
cs = [c_J, c_gs, c_sor]

nmr_iterations = [len(d_J), len(d_gs), len(d_sor)]
names = ["Jacobi", "Gauss–Seidel", "SOR"]
for i, C_i in enumerate(cs):
    print("Testing", names[i])
    print("Converged in:", nmr_iterations[i], "iterations.")
    max_diff = -1
    for i in range(10):
        check = np.max(np.abs(C_i[:, 0] - C_i[:, i]))
        if check > max_diff:
            max_diff = check
    print("Max difference between columns:", max_diff)
    second_diff = C_i[2:, :] - 2*C_i[1:-1, :] + C_i[:-2, :]
    print("The second derivative is given by:", np.max(np.abs(second_diff)))
