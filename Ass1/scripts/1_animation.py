import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def wave_solver_frames(func, L, c, dt, N, T=1.0):
    """
    Generator yielding (t, psi) for the 1D wave equation with fixed ends.
    Uses the same scheme as your code.
    """
    nsteps = int(round(T / dt))
    dx = L / N

    # grid
    x = np.linspace(0.0, L, N + 1)

    # initial condition
    psi0 = np.array([func(xi) for xi in x], dtype=float)
    psi0[0] = 0.0
    psi0[-1] = 0.0

    # first step (assumes initial velocity = 0)
    psi1 = np.zeros_like(psi0)
    psi1[0] = 0.0
    psi1[-1] = 0.0
    for i in range(1, N):
        psi_xx0 = (psi0[i + 1] - 2.0 * psi0[i] + psi0[i - 1]) / (dx * dx)
        psi1[i] = psi0[i] + 0.5 * (c * c) * (dt * dt) * psi_xx0

    psi_prev = psi0
    psi_curr = psi1
    psi_next = np.zeros_like(psi0)

    # yield t=0 and t=dt
    yield 0.0, psi0.copy(), x
    yield dt, psi1.copy(), x

    r2 = (c * dt / dx) ** 2  # CFL factor squared

    for n in range(1, nsteps):
        psi_next[0] = 0.0
        psi_next[-1] = 0.0
        for i in range(1, N):
            lap = psi_curr[i + 1] - 2.0 * psi_curr[i] + psi_curr[i - 1]
            psi_next[i] = 2.0 * psi_curr[i] - psi_prev[i] + r2 * lap

        t = (n + 1) * dt
        yield t, psi_next.copy(), x

        psi_prev, psi_curr = psi_curr, psi_next
        psi_next = np.zeros_like(psi0)

def animate_wave(func, L, c, dt, N, T=1.0, fps=30, ylim=None, save_path=None):
    frames = wave_solver_frames(func, L, c, dt, N, T=T)

    # Grab the first frame to set up axes/line
    t0, psi0, x = next(frames)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    (line,) = ax.plot(x, psi0, lw=2)
    ax.set_title("Finite differences for wave equation (animation)")
    ax.set_xlabel("x")
    ax.set_ylabel("Psi(x,t)")
    ax.grid(True, alpha=0.3)

    if ylim is None:
        # pick a reasonable range from initial condition
        m = max(1e-12, np.max(np.abs(psi0)))
        ax.set_ylim(-1.2 * m, 1.2 * m)
    else:
        ax.set_ylim(*ylim)

    time_text = ax.text(0.02, 0.95, f"t={t0:.3f}", transform=ax.transAxes, va="top")

    # Because we already consumed the first yielded frame, we build a new generator
    def frame_iter():
        yield t0, psi0
        for t, psi, _x in wave_solver_frames(func, L, c, dt, N, T=T):
            yield t, psi

    def init():
        line.set_ydata(psi0)
        time_text.set_text(f"t={t0:.3f}")
        return line, time_text

    def update(frame):
        t, psi = frame
        line.set_ydata(psi)
        time_text.set_text(f"t={t:.3f}")
        return line, time_text

    interval_ms = 1000 / fps
    anim = FuncAnimation(
        fig,
        update,
        frames=frame_iter(),
        init_func=init,
        blit=True,
        interval=interval_ms,
        repeat=True,
    )

    plt.tight_layout()

    if save_path:
        anim.save(save_path, fps=fps)
    else:
        plt.show()

    return anim

if __name__ == "__main__":
    def func(x):
        if 0.2 < x and x < 0.4:
            return np.sin(10.0 * np.pi * x)
        else:
            return 0.0

    animate_wave(func=func, L=1.0, c=1.0, dt=0.001, N=200, T=1.0, fps=60)