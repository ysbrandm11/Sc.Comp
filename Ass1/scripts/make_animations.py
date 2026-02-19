import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Ass1.src.diffusion_td import make_grid, initialize_c, step_explicit


def update(frame, state):
    
    for _ in range(state["steps"]):
        c_old = state["c"]
        c_new = step_explicit(
            c_old,
            D=state["D"],
            dt=state["dt"],
            dx=state["dx"]
        )

        equilibrium_threshold = float(np.max(np.abs(c_new - c_old)))

        state["c"] = c_new
        state["k"] += 1

        if equilibrium_threshold < state["eps"]:
            state["anim"].event_source.stop()
            break

        if state["k"] >= state["max_steps"]:
            state["anim"].event_source.stop()
            break

    state["im"].set_data(state["c"].T)
    t = state["k"] * state["dt"]
    state["ax"].set_title(f"t = {t:.4f} | Equilibrium Threshold = {equilibrium_threshold:.2e}")

    return (state["im"],)


def animate_diffusion(
        N=100,
        L=1.0,
        D=1.0,
        eps=1e-6,
        steps=20,
        max_steps=50000,
        interval_ms=20
):
    _, _, dx = make_grid(N, L)
    dt = 0.9 * dx**2 / (4 * D)

    c = initialize_c(N)

    fig, ax = plt.subplots()
    im = ax.imshow(
        c.T,
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        extent=[0, L, 0, L],
        aspect="equal"
    )

    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    state = {
        "c": c,
        "im": im,
        "ax": ax,
        "D": D,
        "dt": dt,
        "dx": dx,
        "eps": eps,
        "steps": steps,
        "k": 0,
        "max_steps": max_steps,
    }

    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(max_steps // steps + 1),
        fargs=(state,),
        interval=interval_ms,
        repeat=False,
        blit=False
    )
    state["anim"] = anim

    plt.show()


if __name__ == "__main__":
    animate_diffusion()