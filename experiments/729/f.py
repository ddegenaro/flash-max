import torch
from torch import Tensor

c = 1.0

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
    """
    Computes exact E and B fields analytically for the given potential u(t, x, y, z) using PyTorch.
    Optimized for generating training data with a Gaussian base function.
    """
    # Precompute square roots to match the device and dtype of the inputs
    sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=t.dtype, device=t.device))
    sqrt6 = torch.sqrt(torch.tensor(6.0, dtype=t.dtype, device=t.device))

    # 1. Phase arguments
    s1 = sqrt3 * t + x + y + z
    s2 = sqrt3 * t - x + y + z
    s3 = sqrt6 * t - x - 2.0 * y + z

    # Helper function for the second derivative of f(s) = 0.1 * exp(-100 * (s - 0.3)^2)
    # f''(s) = -20 * exp(-100 * (s - 0.3)^2) * (1 - 200 * (s - 0.3)^2)
    def f_double_prime(s: Tensor) -> Tensor:
        s_minus_c = s - 0.3
        s_minus_c_sq = s_minus_c ** 2
        exp_term = torch.exp(-100.0 * s_minus_c_sq)
        return -20.0 * exp_term * (1.0 - 200.0 * s_minus_c_sq)

    # 2. Evaluate the second derivatives
    F1 = f_double_prime(s1)
    F2 = f_double_prime(s2)
    F3 = f_double_prime(s3)

    # 3. Calculate Electric Field components exactly
    # Ex = u_xz - u_yt
    Ex = (1.0 - sqrt3) * F1 - (1.0 + sqrt3) * F2 + (1.0 - 2.0 * sqrt6) * F3

    # Ey = u_yz + u_xt
    Ey = (1.0 + sqrt3) * F1 + (1.0 - sqrt3) * F2 + (2.0 + sqrt6) * F3

    # Ez = u_zz - u_tt
    Ez = -2.0 * F1 - 2.0 * F2 + 5.0 * F3

    # 4. Calculate Magnetic Field components exactly
    # Bx = u_ty + u_xz
    Bx = (sqrt3 + 1.0) * F1 + (sqrt3 - 1.0) * F2 + (2.0 * sqrt6 + 1.0) * F3

    # By = -u_tx + u_yz
    By = (1.0 - sqrt3) * F1 + (1.0 + sqrt3) * F2 + (2.0 - sqrt6) * F3

    # Bz = -u_xx - u_yy
    Bz = -2.0 * F1 - 2.0 * F2 + 5.0 * F3

    # Stack components into a single tensor for E and B fields.
    return torch.vstack((Ex, Ey, Ez, Bx, By, Bz)) / 10.
