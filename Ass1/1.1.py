import numpy as np
import matplotlib.pyplot as plt

phi_0 = 0
phi_L = 0
L = 1
N = 100
dx = L / N
dt = 0.001

def phi_1(x, t):
    return np.sin(2*np.pi*x)

def phi_2(x, t):
    return np.sin(5*np.pi*x)

def phi_3(x,t):
    if 1/5 < x and x < 2/5:
        return np.sin(10*np.pi*x)
    else:
        return 0

def finite_difference_approximation(initial_func, L, N, dt):
    x = np.linspace(0, L, N+1)
    phi = np.zeros((N+1, int(1/dt)+1))

    for i in range(N+1):
        phi[i, 0] = initial_func(x[i], 0)

    for n in range(0, int(1/dt)):
        for i in range(1, N):
            phi[i, n+1] = phi[i, n] + dt * (phi[i+1, n] - 2*phi[i, n] + phi[i-1, n]) / (dx**2)

        phi[0, n+1] = phi_0
        phi[N, n+1] = phi_L

    return x, phi



plt.figure(figsize=(12, 8))
plt.plot(*finite_difference_approximation(phi_1, L, N, dt), label='Initial Condition: sin(2πx)')
plt.show()