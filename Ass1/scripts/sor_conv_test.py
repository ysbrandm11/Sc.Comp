from Solvers import sor
from test_solvers_eq import make_c
import numpy as np
import matplotlib.pyplot as plt



# nmbr_iterations = sor(c.copy(), omega= 1.85, max_iterations= 20000, eps= 1e-5, find_omega=True)

# print(nmbr_iterations)
# #optimal_omega(make_c(Nx=50, Ny=50))

# Define omega values to test

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