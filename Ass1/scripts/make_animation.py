from Wave_function import wave_1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from functions_exc1 import phi_1 as f

x, U, dt = wave_1d(f, N=200, dt=0.001, T=1.0)

fig, ax = plt.subplots()
(line,) = ax.plot(x, U[0, :])
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel(r"$\Psi(x,t)$")
ax.grid(True)

def update(frame):
    line.set_ydata(U[frame, :])
    time_text.set_text(f"t = {frame*dt:.3f}")
    return line, time_text

anim = FuncAnimation(fig, update, frames=len(U),
                     interval=20, blit=True)

plt.show()