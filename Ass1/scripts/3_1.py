import numpy as np

def Jacobi_iteration(A, b, max_iterations=100, eps=1e-6):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.zeros(A.shape[0], dtype=float)
    D = np.diag(A)
    R = A - np.diagflat(D)
    D_inv = D**(-1)

    converged = False
    for k in range(1, max_iterations + 1):
        x_new = D_inv * (b - R @ x)

        if np.linalg.norm(x_new - x) <= eps:
            x = x_new
            converged = True
            break

        x = x_new
    if converged == False:
        print("Warning: Jacobi iteration did not converge within the maximum number of iterations.")

    return x

def gauss_seidel_iterations(A, b, eps=1e-8, max_iterations=10_000):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    n = A.shape[0]
    x = np.zeros(n, dtype=float)

    converged = False
    for k in range(1, max_iterations + 1):
        x_old = x.copy()

        for i in range(n):
            s1 = A[i, :i] @ x[:i]
            s2 = A[i, i+1:] @ x_old[i+1:]
            x[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x - x_old) <= eps:
            converged = True
            break
    if not converged:
        print("Warning: Gauss-Seidel iteration did not converge within the maximum number of iterations.")
    return x

import numpy as np

def SOR(A, b, omega, eps=1e-8, max_iterations=10_000):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]

    x = np.zeros(n, dtype=float)

    converged = False
    for k in range(1, max_iterations + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = A[i, :i] @ x[:i]
            s2 = A[i, i+1:] @ x_old[i+1:]
            x_gs = (b[i] - s1 - s2) / A[i, i]
            x[i] = (1.0 - omega) * x_old[i] + omega * x_gs

        if np.linalg.norm(x - x_old) <= eps:
            converged = True
            break
    if not converged:
        print("Warning: SOR iteration did not converge within the maximum number of iterations.")

    return x

if __name__ == "__main__":
    A = np.array([[4, -1,  0],
                  [-1, 4, -1],
                  [0, -1, 3]], dtype=float)
    b = np.array([15, 10, 10], dtype=float)

    x = SOR(A, b, 1, eps=1e-5, max_iterations=50)
    print("x =", x)