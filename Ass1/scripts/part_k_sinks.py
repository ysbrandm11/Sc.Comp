import numpy as np
import matplotlib.pyplot as plt

from Ass1.src.diffusion_td import make_grid, initialize_c, make_sink_mask
from Ass1.src.diffusion_iter import sor

N = 50
x, y, dx = make_grid(N)
mask = make_sink_mask(N)

c = initialize_c(N, mask=mask)

# sweep w parameter
omegas = np.linspace(1.0, 1.99, 60)
iters = []
for w in omegas:
    it = sor(c.copy(), omega=w, mask=mask, eps=1e-5, max_iterations=20000, find_omega=True)
    iters.append(it)

iters = np.array(iters)
best_idx = np.argmin(iters)
best_w = omegas[best_idx]
print("Best ω with sink:", best_w, "iterations:", iters[best_idx])

c_sol = sor(c.copy(), omega=best_w, mask=mask, eps=1e-5, max_iterations=20000)

# plots
plt.figure()
plt.plot(omegas, iters)
plt.axvline(best_w, linestyle="--")
plt.xlabel("ω")
plt.ylabel("Iterations to convergence")
plt.title(f"SOR iterations vs ω with sink (N={N})")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(c_sol.T, origin="lower", aspect="auto")
plt.colorbar(label="c")
plt.title("Steady concentration with sink object")
plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(mask.T, origin="lower", aspect="auto")
plt.title("Sink mask")
plt.tight_layout()
plt.show()