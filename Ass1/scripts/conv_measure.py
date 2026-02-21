import numpy as np
import matplotlib.pyplot as plt
from Solvers import Jacobi, gauss_seidel, sor

deltas = {}

c_j, d_j = Jacobi(..., return_deltas=True)
deltas["Jacobi"] = d_j

c_gs, d_gs = gauss_seidel(..., return_deltas=True)
deltas["Gauss-Seidel"] = d_gs

for w in [0.8, 1.0, 1.3, 1.7]:
    c_s, d_s = sor(..., omega=w, return_deltas=True)
    deltas[f"SOR ω={w}"] = d_s

plt.figure()
for label, ds in deltas.items():
    plt.semilogy(np.arange(1, len(ds)+1), ds, label=label)
plt.xlabel("Iteration k")
plt.ylabel(r"$\delta_k$")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()