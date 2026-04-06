import torch

c = 1.

sqrt3 = torch.sqrt(torch.tensor(3))
sqrt6 = torch.sqrt(torch.tensor(6))
zero = torch.tensor(0.)

f_func = lambda s : torch.max(zero,torch.min(3.0*s,(1.0-3.0*s)))**3

def u(t, x, y, z):
    """
    Computes E and B fields based on the provided scalar potential mapping.
    Uses central differences for derivatives.
    """
    h = 1e-4  # Step size for numerical differentiation

    # Helper for partial derivatives
    def deriv(func, var_idx, coords):
        c_plus  = [x.clone() if isinstance(x, torch.Tensor) else x for x in coords]
        c_minus = [x.clone() if isinstance(x, torch.Tensor) else x for x in coords]
        c_plus[var_idx]  = c_plus[var_idx]  + h
        c_minus[var_idx] = c_minus[var_idx] - h
        return (func(*c_plus) - func(*c_minus)) / (2 * h)

    def second_deriv(func, var_idx1, var_idx2, coords):
        def perturb(coords, idx, sign):
            c = [x.clone() if isinstance(x, torch.Tensor) else x for x in coords]
            c[idx] = c[idx] + sign * h
            return c

        c_pp = perturb(perturb(coords, var_idx1, +1), var_idx2, +1)
        c_pm = perturb(perturb(coords, var_idx1, +1), var_idx2, -1)
        c_mp = perturb(perturb(coords, var_idx1, -1), var_idx2, +1)
        c_mm = perturb(perturb(coords, var_idx1, -1), var_idx2, -1)

        return (func(*c_pp) - func(*c_pm) - func(*c_mp) + func(*c_mm)) / (4 * h**2)

    # Define u(t, x, y, z) = f(sqrt(3)t + x + y + z)
    U = lambda t, x, y, z: f_func(sqrt3*t + x + y + z)+f_func(sqrt3*t - x + y + z)-f_func(sqrt6*t - x - 2*y + z)

    coords = (t, x, y, z)

    # Calculate components using the provided mapping
    # Indices: t=0, x=1, y=2, z=3
    Ex = second_deriv(U, 1, 3, coords) - second_deriv(U, 2, 0, coords)
    Ey = second_deriv(U, 2, 3, coords) + second_deriv(U, 1, 0, coords)
    Ez = second_deriv(U, 3, 3, coords) - second_deriv(U, 0, 0, coords)

    Bx = second_deriv(U, 0, 2, coords) + second_deriv(U, 1, 3, coords)
    By = -second_deriv(U, 0, 1, coords) + second_deriv(U, 2, 3, coords)
    Bz = -second_deriv(U, 1, 1, coords) - second_deriv(U, 2, 2, coords)

    return torch.vstack((Ex, Ey, Ez, Bx, By, Bz))