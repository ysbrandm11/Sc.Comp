import numpy as np
import matplotlib.pyplot as plt

from Ass1.src.diffusion_td import make_grid, run_diffusion
from Ass1.src.analytic import analytical_c


def main():
    # Parameters
    N = 50
    L = 1.0
    D = 1.0

    _, y, dx = make_grid(N)

    # Picking a dt that is stable (90% of the limit)
    dt_max = dx ** 2 / (4 * D)
    dt = 0.9 * dt_max

    times = [1.0, 0.1, 0.01, 0.001, 0.0]

    results = {}
    for t in times:
        c = run_diffusion(N=N, D=D, dt=dt, t_end=t, L=L)
        
        c_numerical = c.mean(axis=0)
        c_analytical = analytical_c(y, t, D=D, n_terms=200)

        error = abs(c_numerical - c_analytical)
        max_error = np.max(np.abs(error))
        l2_error = np.sqrt(np.mean(error**2))

        results[t] = {
            "c_numerical": c_numerical,
            "c_analytical": c_analytical,
            "error": error
        }

        print(f"t={t:g}  max error={max_error:.3e}  L2 error={l2_error:.3e}")


    plt.figure()
    
    for t in times:
        plt.plot(y, results[t]["c_numerical"], marker="o", linestyle="", label=f"num t={t:g}")
        plt.plot(y, results[t]["c_analytical"], linestyle="-", label=f"ana t={t:g}")

    plt.xlabel("y")
    plt.ylabel("c(y,t)")
    plt.title("Time-dependent diffusion: numerical vs analytical")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Ass1/results/diffusion_td_validation_profiles.png", dpi=200)
    

    plt.figure()

    for t in times:
        plt.plot(y, results[t]["error"], label=f"t={t:g}")

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("y")
    plt.ylabel("error")
    plt.title("Absolute Error: abs(numerical - analytical)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Ass1/results/diffusion_td_error.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()