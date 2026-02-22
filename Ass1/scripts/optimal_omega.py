import numpy as np
import matplotlib.pyplot as plt
from Ass1.src.diffusion_td import initialize_c
from Ass1.src.diffusion_iter import sor

def best_omega_for_N(N, eps=1e-5, max_iterations=20000, n_omegas=20):
    c0 = initialize_c(N)
    omegas = np.linspace(1.0, 1.99, n_omegas)

    iterations = np.empty_like(omegas)
    for i, w in enumerate(omegas):
        iterations[i] = sor(
            c0.copy(),
            omega=w,
            eps=eps,
            max_iterations=max_iterations,
            find_omega=True
        )

    best_idx = np.argmin(iterations)
    return omegas[best_idx], int(iterations[best_idx]), omegas, iterations


Ns = [10, 20, 30, 40, 50]

best_ws = []
best_its = []

for N in Ns:
    w_star, it_star, _, _ = best_omega_for_N(N, eps=1e-5, n_omegas=60)
    best_ws.append(w_star)
    best_its.append(it_star)
    print(f"N={N:3d}  ω={w_star:.4f}  iterations={it_star}")

plt.figure()
plt.plot(Ns, best_ws, marker="o")
plt.xlabel("Grid size N")
plt.ylabel("Optimal ω")
plt.title("Optimal SOR factor ω vs N")
plt.grid(True)
plt.tight_layout()
plt.show()