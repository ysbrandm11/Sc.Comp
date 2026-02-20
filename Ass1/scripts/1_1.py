import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation



def phi_1(x):
    return np.sin(2.0 * np.pi * x)

def phi_2(x):
    return np.sin(5.0 * np.pi * x)

def phi_3(x):
    if 0.2 < x and x < 0.4:
        return np.sin(10.0 * np.pi * x)
    else:
        return 0.0

def main(func, L, c, dt, N, plot_times):
    T = 1.0
    nsteps = int(round(T / dt))
    dx = L / N
    plot_steps = {}
    for t in plot_times:
        plot_steps[int(round(t / dt))] = t
    x = np.zeros(N + 1)
    for i in range(N + 1):
        x[i] = i * dx
    psi0 = np.zeros(N + 1)
    for i in range(N + 1):
        psi0[i] = func(x[i])
    psi0[0] = 0.0
    psi0[N] = 0.0

    psi1 = np.zeros(N + 1)
    psi1[0] = 0.0
    psi1[N] = 0.0

    for i in range(1, N):
        psi_xx0 = (psi0[i+1] - 2.0*psi0[i] + psi0[i-1]) / (dx**2)
        psi1[i] = psi0[i] + 0.5 * (c*c) * (dt*dt) * psi_xx0

    psi_prev = psi0.copy()
    psi_curr = psi1.copy()
    psi_next = np.zeros(N + 1)

    snapshots = []
    snapshots.append((0.0, psi0.copy()))
    if 1 in plot_steps:
        snapshots.append((plot_steps[1], psi1.copy()))

    for n in range(1, nsteps):
        psi_next[0] = 0.0
        psi_next[N] = 0.0
        for i in range(1, N):
            lap = psi_curr[i+1] - 2.0*psi_curr[i] + psi_curr[i-1]
            psi_next[i] = 2.0*psi_curr[i] - psi_prev[i] + (c*dt/dx)**2 * lap
        step = n + 1
        if step in plot_steps:
            snapshots.append((plot_steps[step], psi_next.copy()))

        psi_prev, psi_curr = psi_curr, psi_next
        psi_next = np.zeros(N + 1)

    plt.figure(figsize=(8, 4.5))
    for t, s in snapshots:
        plt.plot(x, s, label=f"t={t:.3f}")

    plt.title("Finite differences for wave equation (case i)")
    plt.xlabel("x")
    plt.ylabel("Psi(x,t)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

L = 1.0
c = 1.0
dt = 0.001
N = 200
plot_times = [0.0, 0.35, 0.7, 1]
main(phi_1, L, c, dt, N, plot_times)



def wave_solver_frames(func, L, c, dt, N, T=1.0):

    nsteps = int(T/dt)
    dx = L / N
    x = np.linspace(0.0, L, N + 1)

    psi0 = np.array([func(xi) for xi in x], dtype=float)
    psi0[0] = 0.0
    psi0[-1] = 0.0

    psi1 = np.zeros_like(psi0)
    psi1[0] = 0.0
    psi1[-1] = 0.0
    for i in range(1, N):
        psi_xx0 = (psi0[i + 1] - 2.0 * psi0[i] + psi0[i - 1]) / (dx * dx)
        psi1[i] = psi0[i] + 0.5 * (c * c) * (dt * dt) * psi_xx0

    psi_prev = psi0
    psi_curr = psi1
    psi_next = np.zeros_like(psi0)

    yield 0.0, psi0.copy(), x
    yield dt, psi1.copy(), x

    r2 = (c * dt / dx) ** 2

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
        m = max(1e-12, np.max(np.abs(psi0)))
        ax.set_ylim(-1.2 * m, 1.2 * m)
    else:
        ax.set_ylim(*ylim)

    time_text = ax.text(0.02, 0.95, f"t={t0:.3f}", transform=ax.transAxes, va="top")

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
    animate_wave(func=phi_1, L=1.0, c=1.0, dt=0.001, N=200, T=1.0, fps=60)