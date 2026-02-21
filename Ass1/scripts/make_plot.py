import matplotlib.pyplot as plt
import numpy as np
from Wave_function import wave_1d

# Change according to which function wanted
from functions_exc1 import phi_1 as f

x, U, dt = wave_1d(f, N=200, dt=0.001, T=1.0)
times = [0.0, 0.1, 0.2, 0.3, 0.4]
plt.figure()

for t in times:
    n = int(t / dt)   # convert time → time index
    plt.plot(x, U[n, :], label=f"t={t}")

plt.xlabel("x")
plt.ylabel(r"$\Psi(x,t)$")
plt.title("Time development of the string")
plt.legend()
plt.grid(True)
plt.show()