import numpy as np
from Ass1.src.diffusion_iter import jacobi, gauss_seidel, sor
from Ass1.src.diffusion_td import make_grid, initialize_c

N = 50
x, y, dx = make_grid(N, L=1.0)
c = initialize_c(N)

c_J, d_J= jacobi(c.copy(), max_iterations=5000, eps=1e-5, return_delta=True)
c_gs, d_gs = gauss_seidel(c.copy(), max_iterations=5000, eps=1e-5, return_delta=True)
c_sor, d_sor = sor(c.copy(), omega=1.85, max_iterations=5000, eps=1e-5, return_delta=True)

cs = [c_J, c_gs, c_sor]
deltas = [d_J, d_gs, d_sor]
names = ["Jacobi", "Gauss–Seidel", "SOR"]

for idx, C_i in enumerate(cs):
    print("\nTesting", names[idx])
    print("Converged in:", len(deltas[idx]), "iterations.")
    print("Last delta:", deltas[idx][-1])
    
    max_diff = 0.0
    for i in range(1, C_i.shape[0]):
        check = np.max(np.abs(C_i[i, :] - C_i[0, :]))
        if check > max_diff:
            max_diff = check
    print("Max variation across x:", max_diff)

    C_exact = np.tile(y, (C_i.shape[0], 1))
    max_err = np.max(np.abs(C_i - C_exact))
    print("Max error vs c(y)=y:", max_err)

    second_diff_y = C_i[:, 2:] - 2*C_i[:, 1:-1] + C_i[:, :-2]
    print("The second derivative is given by:", np.max(np.abs(second_diff_y)))


