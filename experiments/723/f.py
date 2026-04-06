import torch
from torch import Tensor

c = 1.0

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
    """
    Computes exact E and B fields analytically for the given potential u(t, x, y, z) using PyTorch.
    This provides zero truncation error and is highly optimized for generating training data.
    """
    # Precompute square roots as tensors to match the device and dtype of the inputs.
    # This prevents errors if your inputs are on a GPU or use a specific precision (e.g., float64).
    sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=t.dtype, device=t.device))
    sqrt6 = torch.sqrt(torch.tensor(6.0, dtype=t.dtype, device=t.device))

    # 1. Phase arguments
    s1 = sqrt3 * t + x + y + z
    s2 = sqrt3 * t - x + y + z
    s3 = sqrt6 * t - x - 2.0 * y + z

    # 2. Second derivative of f(s) = cos(s) is f''(s) = -cos(s)
    F1 = -torch.cos(s1)
    F2 = -torch.cos(s2)
    F3 = -torch.cos(s3)

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
    # If t, x, y, z are shape (N,), E_field and B_field will be shape (3, N).
    return torch.vstack((Ex, Ey, Ez, Bx, By, Bz))