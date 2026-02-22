import matplotlib.pyplot as plt

from Ass1.src.wave import phi_1, phi_2, phi_3
from Ass1.src.wave_solver import wave_1d

# Change according to which function wanted
f = phi_3

N = 200
dt = 0.001
T = 1.0

x, U, dt = wave_1d(f, N=200, dt=0.001, T=1.0)

times = [0.0, 0.1, 0.2, 0.3, 0.4]
plt.figure()

for t in times:
    n = int(t / dt)
    plt.plot(x, U[n, :], label=f"t={t}")

plt.xlabel("x")
plt.ylabel(r"$\Psi(x,t)$")
plt.title("Time development of the string")
plt.legend()
plt.grid(True)
plt.savefig("string_time_development_phi2.png")
plt.show()