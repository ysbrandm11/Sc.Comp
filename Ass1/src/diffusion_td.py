import numpy as np

def make_grid(N: int , L: float = 1.0):
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N+1)
    return x, y, dx


def initialize_c(N: int) -> np.ndarray:
    c = np.zeros((N, N+1), dtype=float)
    apply_bc(c)
    return c


def apply_bc(c: np.ndarray) -> None:
    c[:, 0] = 0.0
    c[:, -1] = 1.0


def step_explicit(c: np.ndarray, D: float, dt: float, dx: float) -> np.ndarray:
    N, Ny = c.shape
    c_next = c.copy()

    # Roll the neighbors 
    c_ip1 = np.roll(c, -1, axis=0)
    c_im1 = np.roll(c, 1, axis=0)

    c_next[:, 1:-1] = (
        c[:, 1:-1] 
        + (D * dt / dx**2) * (
            c_ip1[:, 1:-1]
            + c_im1[:, 1:-1]
            + c[:, 2:]
            + c[:, :-2]
            - 4 * c[:, 1:-1]
        )
    )
    
    apply_bc(c_next)
    return c_next


def run_diffusion(N: int, D: float, dt: float, t_end: float, L: float = 1.0) -> np.ndarray:
    x, y, dx = make_grid(N, L)
    c = initialize_c(N)

    # Safety check to make sure scheme is stable
    if 4 * D * dt / dx ** 2 > 1:
        raise ValueError("Unstable parameters: It must hold that 4*D*dt/dx**2 <= 1")

    n_steps = int(t_end / dt)

    for _ in range(n_steps):
        c = step_explicit(c, D, dt, dx)

    return c