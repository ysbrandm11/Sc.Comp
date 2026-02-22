import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Ass1.src.wave_solver import wave_1d
from Ass1.src.wave import phi_1, phi_2, phi_3

# Change according to which function wanted
f = phi_1

N = 200
dt = 0.001
T = 1.0

x, U, dt = wave_1d(f, N=200, dt=0.001, T=1.0)

fig, ax = plt.subplots()
(line,) = ax.plot(x, U[0, :])
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

ax.set_xlim(0, 1)

umin = np.min(U)
umax = np.max(U)
ax.set_ylim(umin, umax)

ax.set_xlabel("x")
ax.set_ylabel(r"$\Psi(x,t)$")
ax.grid(True)

def update(frame):
    line.set_ydata(U[frame, :])
    time_text.set_text(f"t = {frame*dt:.3f}")
    return line, time_text

anim = FuncAnimation(
    fig,
    update,
    frames=U.shape[0],
    interval=20,
    blit=True)

plt.show()