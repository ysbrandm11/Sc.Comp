from Solvers import sor
from test_solvers_eq import make_c
import numpy as np
import matplotlib.pyplot as plt

c = make_c(Nx=21, Ny=21)
omegas = np.linspace(1.7, 1.98, 20)   # avoid 2.0 (can diverge)

iteration_counts = []

for w in omegas:
    nmbr_iterations = sor(
        c.copy(),
        omega=w,
        max_iterations=20000,
        eps=1e-5,
        find_omega=True
    )
    iteration_counts.append(nmbr_iterations)

iteration_counts = np.array(iteration_counts)

plt.figure()
plt.plot(omegas, iteration_counts)
plt.xlabel("Omega (ω)")
plt.ylabel("Iterations until convergence")
plt.title("SOR convergence vs relaxation parameter")
plt.grid(True)
plt.show()