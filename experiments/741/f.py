import torch
from torch import Tensor

c = 1.0

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """
    Computes exact E and B fields analytically for the given potential u(t, x, y, z) using PyTorch.
    Optimized for generating training data with a wider Gaussian base function.
    """
    sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=t.dtype, device=t.device))
    sqrt6 = torch.sqrt(torch.tensor(6.0, dtype=t.dtype, device=t.device))

    s1 = sqrt3 * t + x + y + z
    s2 = sqrt3 * t - x + y + z
    s3 = sqrt6 * t - x - 2.0 * y + z

    # Updated helper function for f(s) = 0.1 * exp(-10 * (s - 0.3)^2)
    # f''(s) = -2 * exp(-10 * (s - 0.3)^2) * (1 - 20 * (s - 0.3)^2)
    def f_double_prime(s: Tensor) -> Tensor:
        s_minus_c = s - 0.3
        s_minus_c_sq = s_minus_c ** 2
        exp_term = torch.exp(-10.0 * s_minus_c_sq)
        return -2.0 * exp_term * (1.0 - 20.0 * s_minus_c_sq)

    F1 = f_double_prime(s1)
    F2 = f_double_prime(s2)
    F3 = f_double_prime(s3)

    Ex = (1.0 - sqrt3) * F1 - (1.0 + sqrt3) * F2 + (1.0 - 2.0 * sqrt6) * F3
    Ey = (1.0 + sqrt3) * F1 + (1.0 - sqrt3) * F2 + (2.0 + sqrt6) * F3
    Ez = -2.0 * F1 - 2.0 * F2 + 5.0 * F3

    Bx = (sqrt3 + 1.0) * F1 + (sqrt3 - 1.0) * F2 + (2.0 * sqrt6 + 1.0) * F3
    By = (1.0 - sqrt3) * F1 + (1.0 + sqrt3) * F2 + (2.0 - sqrt6) * F3
    Bz = -2.0 * F1 - 2.0 * F2 + 5.0 * F3

    return 0.1*torch.vstack((Ex, Ey, Ez, Bx, By, Bz))