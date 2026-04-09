import torch
from torch import Tensor

c = 1.0

# Hopf vibration

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """
    Computes the E and B fields for the electromagnetic knot.
    """
    # Grouping repeated terms for computational efficiency
    R2_minus_t2 = 1 + x**2 + y**2 + z**2 - t**2
    denom = (R2_minus_t2**2 + 4*t**2)**3

    term1 = R2_minus_t2**3 - 12 * t**2 * R2_minus_t2
    term2 = 8 * t**3 - 6 * t * R2_minus_t2**2

    # Electric Field Components
    Ex = (term1 * ((t - z)**2 - 1 - x**2 + y**2) - term2 * (2*x*y - 2*(t - z))) / denom
    Ey = (term1 * (-2*x*y - 2*(t - z)) - term2 * (1 - (t - z)**2 - x**2 + y**2)) / denom
    Ez = (term1 * (2*x*(t - z) - 2*y) + term2 * (2*x + 2*y*(t - z))) / denom

    # Magnetic Field Components
    Bx = (term2 * ((t - z)**2 - 1 - x**2 + y**2) + term1 * (2*x*y - 2*(t - z))) / denom
    By = (term2 * (-2*x*y - 2*(t - z)) + term1 * (1 - (t - z)**2 - x**2 + y**2)) / denom
    Bz = (term2 * (2*x*(t - z) - 2*y) - term1 * (2*x + 2*y*(t - z))) / denom

    return torch.vstack((Ex, Ey, Ez, Bx, By, Bz))
