from Solvers import sor
from test_solvers_eq import make_c

omega_max = 2.0
omega_min = 1.7

def find_max_omega(c, omega_min=omega_min, omega_max=omega_max, eps=1e-3):
    #val_min = sor(c.copy(), omega=omega_min, max_iterations=int(1e5), find_omega=True)
    val_max = sor(c.copy(), eps = 0.01, omega=omega_max, max_iterations=int(1e5), find_omega=True)
    dif = omega_max - omega_min
    print(dif)
    while dif > eps:
        omega_mid = (omega_min + omega_max) / 2
        val_mid = sor(c.copy(), eps=0.01, omega=omega_mid, max_iterations=int(1e5), find_omega=True)
        if val_mid >= val_max:
            omega_max = omega_mid
            val_max = val_mid
        else:
            omega_min = omega_mid
            #val_min = val_mid
        dif = omega_max - omega_min
        print(dif)
    return omega_mid

find_max_omega(make_c(Nx=50, Ny=50))