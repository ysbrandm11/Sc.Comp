import numpy as np
import matplotlib.pyplot as plt

from Ass1.src.diffusion_iter import jacobi, gauss_seidel, sor
from Ass1.src.diffusion_td import make_grid, initialize_c

N = 50
x, y, dx = make_grid(N, L=1.0)
c = initialize_c(N)

# Jacobi + GS
_, dJ  = jacobi(c.copy(), max_iterations=5000, eps=1e-5, return_delta=True)
_, dGS = gauss_seidel(c.copy(), max_iterations=5000, eps=1e-5, return_delta=True)

# SOR for multiple omega
omegas = [1.0, 1.5, 1.7, 1.85, 1.9]
dSOR = {}
for w in omegas:
    _, dw = sor(c.copy(), omega=w, max_iterations=5000, eps=1e-5, return_delta=True)
    dSOR[w] = dw

plt.figure()
plt.yscale("log")

plt.semilogy(dJ, label="Jacobi")
plt.semilogy(dGS, label="Gauss–Seidel")

for w, dw in dSOR.items():
    plt.plot(range(1, len(dw)+1), dw, label=f"SOR ω={w}")

plt.xlabel("Iteration k")
plt.ylabel(r"$\delta_k = \max |c^{k+1}-c^k|$")
plt.title(f"Convergence comparison (N={N})")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()