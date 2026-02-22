import numpy as np

def wave_1d(f, N=200, dt=0.001, c=1.0, T=1.0):
    L = 1.0
    dx = L / N
    Nt = int(T / dt)
    x = np.linspace(0, L, N+1)
    U = np.zeros((Nt+1, N+1))

    U[0, :] = f(x)
    U[0, 0] = 0.0
    U[0, -1] = 0.0

    for i in range(1, N):
        U[1, i] = (
            U[0, i]
            + 0.5 * (c * dt / dx)**2 *
            (U[0, i+1] - 2*U[0, i] + U[0, i-1])
        )

    for n in range(1, Nt):
        for i in range(1, N):
            U[n+1, i] = (
                2*U[n, i]
                - U[n-1, i]
                + (c * dt / dx)**2 *
                (U[n, i+1] - 2*U[n, i] + U[n, i-1])
            )

    return x, U, dt