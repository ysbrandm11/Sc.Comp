import numpy as np
from Ass1.src.diffusion_td import apply_bc

def jacobi(c, max_iterations=1000, eps=1e-5, return_delta=False):
    c = np.array(c, dtype=float, copy=True)
    c_new = c.copy()
    deltas = []

    for k in range(max_iterations):
        c_ip1 = np.roll(c, -1, axis=0)
        c_im1 = np.roll(c,  1, axis=0)

        c_new[:, 1:-1] = 0.25 * (
            c_ip1[:, 1:-1] +
            c_im1[:, 1:-1] +
            c[:, 2:] +
            c[:, :-2]
        )

        apply_bc(c_new)      

        delta = np.max(np.abs(c_new[:, 1:-1] - c[:, 1:-1]))
        deltas.append(delta)

        if delta < eps:
            c = c_new
            print(f"Converged after {k+1} iterations")
            break

        c, c_new = c_new, c

    if return_delta:
        return (c, deltas)
    else:
        return c
    
def gauss_seidel(c, max_iterations=1000, eps=1e-5, return_delta=False):
    c = np.array(c, dtype=float, copy=True)
    nx, ny = c.shape
    deltas = []

    for k in range(max_iterations):
        delta = 0.0

        for j in range(1, ny - 1):
            for i in range(nx):
                c_ip1 = (i + 1) % nx
                c_im1 = (i - 1) % nx

                old_value = c[i, j]
                c[i, j] = 0.25 * (
                    c[c_ip1, j] +
                    c[c_im1, j] +
                    c[i, j + 1] +
                    c[i, j - 1]
                )

                diff = abs(c[i, j] - old_value)
                if diff > delta:
                    delta = diff

        apply_bc(c)
        deltas.append(delta)

        if delta < eps:
            print(f"Converged after {k+1} iterations")
            break

    if return_delta:
        return (c, deltas)
    else:
        return c

def sor(c, omega, max_iterations=1000, eps=1e-5, return_delta = False,
        find_omega=False):
    c = np.array(c, dtype=float, copy=True)
    nx, ny = c.shape
    deltas = []
    converged = False

    for k in range(max_iterations):
        delta = 0.0

        for j in range(1, ny - 1):
            for i in range(nx):
                c_ip1 = (i + 1) % nx
                c_im1 = (i - 1) % nx
                
                old_value = c[i, j]
                gs = 0.25 * (
                    c[c_ip1, j] +
                    c[c_im1, j] +
                    c[i, j + 1] +
                    c[i, j - 1]
                )
                c[i, j] = (1.0 - omega) * old_value + omega * gs

                diff = abs(c[i, j] - old_value)
                if diff > delta:
                    delta = diff

        apply_bc(c)
        deltas.append(delta)

        if delta < eps:
            converged = True
            print(f"Converged after {k+1} iterations")
            break
    
    if find_omega:
        if converged:
            return k+1
        else:
            return max_iterations
        
    if return_delta:
        return (c, deltas)
    else:
        return c