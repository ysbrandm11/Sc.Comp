import numpy as np

def jacobi(c, max_iterations=1000, eps=1e-6, return_delta = False):
    """
    Jacobi iteration for Laplace equation on a 2D grid.
    Boundary values are assumed fixed.
    """
    c = c.astype(float)
    nx, ny = c.shape
    c_new = c.copy()
    deltas = []
    for k in range(max_iterations):
        delta = 0.0
        # loop over interior points only
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                c_new[i, j] = 0.25 * (
                    c[i+1, j] +
                    c[i-1, j] +
                    c[i, j+1] +
                    c[i, j-1]
                )
                delta = max(delta, abs(c_new[i, j] - c[i, j]))
                deltas.append(delta)

        if delta < eps:
            print(f"Converged after {k+1} iterations")
            break
        c[:] = c_new[:]

    return c_new if not return_delta else (c_new, deltas)

def gauss_seidel(c, max_iterations=1000, eps=1e-6, return_delta = False):
    c = c.astype(float)
    nx, ny = c.shape
    deltas = []
    for k in range(max_iterations):
        delta = 0.0

        for i in range(1, nx-1):
            for j in range(1, ny-1):

                old_value = c[i, j]

                c[i, j] = 0.25 * (
                    c[i+1, j] +   # old
                    c[i-1, j] +   # already updated
                    c[i, j+1] +   # old
                    c[i, j-1]     # already updated
                )

                delta = max(delta, abs(c[i, j] - old_value))
                deltas.append(delta)

        if delta < eps:
            print(f"Converged after {k+1} iterations")
            break

    return c if not return_delta else (c, deltas)

def sor(c, omega, max_iterations=1000, eps=1e-6, return_delta = False):
    c = c.astype(float)
    nx, ny = c.shape
    deltas = []
    for k in range(max_iterations):
        delta = 0.0

        for i in range(1, nx-1):
            for j in range(1, ny-1):

                old_value = c[i, j]

                c[i, j] = omega/4 * (
                    c[i+1, j] +   # old
                    c[i-1, j] +   # already updated
                    c[i, j+1] +   # old
                    c[i, j-1]     # already updated
                ) + (1 - omega) * old_value

                delta = max(delta, abs(c[i, j] - old_value))
                deltas.append(delta)

        if delta < eps:
            print(f"Converged after {k+1} iterations")
            break
    return c if not return_delta else (c, deltas)