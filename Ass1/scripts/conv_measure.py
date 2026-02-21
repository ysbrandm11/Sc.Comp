from test_solvers_eq import make_c
import matplotlib.pyplot as plt
from Solvers import jacobi, gauss_seidel, sor
import numpy as np

Nx, Ny = 50, 50
c0 = make_c(Nx, Ny)
c0[1:-1, :] = np.random.rand(c0.shape[0]-2, c0.shape[1])

_, d_jac = jacobi(c0.copy(), return_delta=True, max_iterations=100000, eps=1e-12)
_, d_gs  = gauss_seidel(c0.copy(), return_delta=True, max_iterations=10000, eps=1e-12)

print("Jacobi iterations:", len(d_jac), "final δ:", d_jac[-1])
print("GS iterations:    ", len(d_gs), "final δ:", d_gs[-1])

print("klaar met jacobi en gauss-seidel")
# omegas = [0.8, 1.0, 1.3]
# sor_deltas = {}
# for w in omegas:
#     _, d = sor(c0.copy(), omega=w, return_delta=True, max_iterations=500, eps=1e-12)
#     print(f"klaar met SOR ω={w}")
#     sor_deltas[w] = d


print("len(dJ), len(dG):", len(d_jac), len(d_gs))
print("max |dJ-dG|:", np.max(np.abs(np.array(d_jac) - np.array(d_gs))))
print("allclose?:", np.allclose(d_jac, d_gs))
# Plot (log-lin)
plt.figure()
plt.semilogy(d_jac, label="Jacobi")
plt.semilogy(d_gs,  label="Gauss–Seidel")
# for w, d in sor_deltas.items():
#     plt.semilogy(d, label=f"SOR ω={w}")

plt.xlabel("Iteration k")
plt.ylabel("δ(k)")
plt.legend()
plt.grid(True, which="both")
plt.show()