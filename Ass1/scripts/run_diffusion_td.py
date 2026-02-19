import numpy as np
import matplotlib.pyplot as plt

from Ass1.src.diffusion_td import make_grid, run_diffusion
from Ass1.src.analytic import analytical_c


def plot_field(c: np.ndarray, t: float, L: float, path: str) -> None:
    plt.figure()
    plt.imshow(
        c.T,
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        extent=[0, L, 0, L],
        aspect="equal"
        )
    plt.colorbar(label="c(x,y,t)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Concentration field at t={t:g}")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    # Parameters
    N = 100
    L = 1.0
    D = 1.0

    _, y, dx = make_grid(N, L)

    # Picking a dt that is stable (90% of the limit)
    dt_max = dx ** 2 / (4 * D)
    dt = 0.9 * dt_max

    # Times in decreasing order because it looks better when plotted
    times = [1.0, 0.1, 0.01, 0.001, 0.0]

    results = {}
    for t in times:
        c = run_diffusion(N=N, D=D, dt=dt, t_end=t, L=L)

        t_tag = f"{t:.3g}".replace(".", "p")
        plot_field(c, t, L, f"Ass1/results/diffusion_td_field_t{t_tag}.png")
        
        c_numerical = c.mean(axis=0)
        c_analytical = analytical_c(y, t, D=D, n_terms=200)

        error = np.abs(c_numerical - c_analytical)
        max_error = np.max(error)
        l2_error = np.sqrt(np.mean(error**2))

        results[t] = {
            "c_numerical": c_numerical,
            "c_analytical": c_analytical,
            "error": error
        }

        print(f"t={t:g}  max error={max_error:.3e}  L2 error={l2_error:.3e}")


    # Numerical vs Analytical plots
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

    # Error plot
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