import torch
from torch import Tensor

c = 1.0

# Radial waves

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """
    Computes exact E and B fields analytically for a spherical potential u(t, x, y, z) = f(r-t)/r.
    Adapted for f(s) = 0.01 * exp(-10 * (s - 0.7)^2).
    """
    # 1. Calculate radius (clamped to avoid division by zero at the origin)
    r = torch.sqrt(x**2 + y**2 + z**2).clamp(min=1e-5)

    # 2. Phase argument
    s = r - t
    s_mc = s - 0.7  # <--- UPDATED SHIFT HERE
    s_mc_sq = s_mc**2

    # 3. Base function f(s) and its derivatives
    exp_term = torch.exp(-10.0 * s_mc_sq)
    f = 0.01 * exp_term
    f_p = -0.2 * s_mc * exp_term
    f_pp = -0.2 * exp_term * (1.0 - 20.0 * s_mc_sq)

    # 4. Helper scalar fields (g, h, q) based on analytical differentiation
    g = f_p / (r**2) - f / (r**3)
    h = f_pp / (r**3) - 3.0 * f_p / (r**4) + 3.0 * f / (r**5)
    q = -f_pp / (r**2) + f_p / (r**3)

    # 5. Calculate Electric Field components
    Ex = x * z * h - y * q
    Ey = y * z * h + x * q
    Ez = g + (z**2) * h - f_pp / r

    # 6. Calculate Magnetic Field components
    Bx = y * q + x * z * h
    By = -x * q + y * z * h
    Bz = -2.0 * g - (x**2 + y**2) * h

    return 10.0 * torch.vstack((Ex, Ey, Ez, Bx, By, Bz))
