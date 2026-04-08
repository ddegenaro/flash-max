import torch
import math
from torch import Tensor

c = 1.0

_k = torch.randn((100, 3)) * math.sqrt(0.1)
_b = torch.randn(100)
_omega = torch.sqrt(_k[:, 0]**2 + _k[:, 1]**2 + _k[:, 2]**2)

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    shape = t.shape
    t_f, x_f, y_f, z_f = t.reshape(-1), x.reshape(-1), y.reshape(-1), z.reshape(-1)

    s = (_omega.unsqueeze(1) * t_f.unsqueeze(0) +
         _k[:, 0].unsqueeze(1) * x_f.unsqueeze(0) +
         _k[:, 1].unsqueeze(1) * y_f.unsqueeze(0) +
         _k[:, 2].unsqueeze(1) * z_f.unsqueeze(0) +
         _b.unsqueeze(1))

    s_minus_c = s - 0.3
    s_minus_c_sq = s_minus_c ** 2
    F = -0.2 * torch.exp(-10.0 * s_minus_c_sq) * (1.0 - 20.0 * s_minus_c_sq)

    k1, k2, k3 = _k[:, 0:1], _k[:, 1:2], _k[:, 2:3]
    w = _omega.unsqueeze(1)

    Ex = torch.sum((k1 * k3 - k2 * w) * F, dim=0).view(shape)
    Ey = torch.sum((k2 * k3 + k1 * w) * F, dim=0).view(shape)
    Ez = torch.sum((k3**2 - w**2)       * F, dim=0).view(shape)

    Bx = torch.sum((w * k2 + k1 * k3)   * F, dim=0).view(shape)
    By = torch.sum((-w * k1 + k2 * k3)  * F, dim=0).view(shape)
    Bz = torch.sum((-k1**2 - k2**2)     * F, dim=0).view(shape)

    return torch.vstack((Ex, Ey, Ez, Bx, By, Bz))