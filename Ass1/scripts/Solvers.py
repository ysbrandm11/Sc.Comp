import numpy as np

def jacobi(c, max_iterations=1000, eps=1e-5, return_delta=False):
    c = c.astype(float)
    ny, nx = c.shape
    c_new = c.copy()
    deltas = []

    for k in range(max_iterations):
        delta = 0.0
        for j in range(1, ny - 1):
            for i in range(nx):

                c_new[j, i] = 0.25 * (
                    c[j, (i + 1) % nx] +
                    c[j, (i - 1) % nx] +
                    c[j + 1, i] +
                    c[j - 1, i]
                )

                delta = max(delta, abs(c_new[j, i] - c[j, i]))
        c_new[0, :]  = c[0, :]
        c_new[-1, :] = c[-1, :]

        deltas.append(delta)

        if delta < eps:
            print(f"Converged after {k+1} iterations")
            break
        c[:, :] = c_new[:, :]

    return (c_new, deltas) if return_delta else c_new

def gauss_seidel(c, max_iterations=1000, eps=1e-5, return_delta=False):
    c = c.astype(float)
    ny, nx = c.shape
    deltas = []

    # store fixed y-boundaries (Dirichlet)
    bottom = c[0, :].copy()
    top    = c[-1, :].copy()

    for k in range(max_iterations):
        delta = 0.0

        for j in range(1, ny - 1):
            for i in range(nx):

                old_value = c[j, i]
                c[j, i] = 0.25 * (
                    c[j, (i + 1) % nx] +
                    c[j, (i - 1) % nx] +
                    c[j + 1, i] +
                    c[j - 1, i]
                )

                delta = max(delta, abs(c[j, i] - old_value))

        c[0, :]  = bottom
        c[-1, :] = top

        deltas.append(delta)

        if delta < eps:
            print(f"Converged after {k+1} iterations")
            break

    return (c, deltas) if return_delta else c

def sor(c, omega, max_iterations=1000, eps=1e-6, return_delta = False):
    c = c.astype(float)
    nx, ny = c.shape
    deltas = []
    bottom = c[0, :].copy()
    top    = c[-1, :].copy()

    for k in range(max_iterations):
        delta = 0.0

        for j in range(1, ny - 1):
            for i in range(nx):

                old_value = c[j, i]
                c[j, i] = (omega/4) * (
                    c[j, (i + 1) % nx] +
                    c[j, (i - 1) % nx] +
                    c[j + 1, i] +
                    c[j - 1, i]
                ) + (1 - omega) * old_value
                delta = max(delta, abs(c[j, i] - old_value))

        c[0, :]  = bottom
        c[-1, :] = top

        deltas.append(delta)

        if delta < eps:
            print(f"Converged after {k+1} iterations")
            break
    return c if not return_delta else (c, deltas)