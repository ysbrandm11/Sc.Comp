import numpy as np

def initial_condition(x, case):
    if case == 1:
        return np.sin(2 * np.pi * x)
    if case == 2:
        return np.sin(5 * np.pi * x)
    if case == 3:
        psi = np.zeros_like(x)
        set = (x > 1/5) & (x < 2/5)
        psi[set] = np.sin(5 * np.pi * x[set])
        return psi


def string_simulation(case, N=100, dt=0.001, T=1, c=1, L=1):
    dx = L / N
    a = c * dt / dx

    x = np.linspace(0, L, N + 1)    # N = no. intervals, N+1 = no. nodes

    psi_prev = initial_condition(x, case)
    psi_prev[0] = 0         # bound. condition
    psi_prev[-1] = 0        # bound. condition

    psi_curr = np.zeros(N+1)

    for i in range(1, N):
        psi_curr[i] = psi_prev[i] + 0.5 * (a ** 2) * (psi_prev[i + 1] - 2 * psi_prev[i] + psi_prev[i - 1])

    psi_curr[0] = 0
    psi_curr[-1] = 0

    nt = int(T / dt)

    for n in range(1, nt):
        psi_next = np.zeros(N+1)

        for i in range(1, N):
            psi_next[i] = 2 * psi_curr[i] - psi_prev[i] + (a ** 2) * (psi_curr[i + 1] - 2 * psi_curr[i] + psi_curr[i - 1])

        psi_next[0] = 0.0
        psi_next[-1] = 0.0

        psi_prev = psi_curr
        psi_curr = psi_next


cases = [1, 2, 3]

for i in cases:
    string_simulation(i)