from math import sqrt

import torch
from torch import Tensor

c = 1.0

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    
    num_modes = 100
    k = torch.randn((num_modes, 3)) * sqrt(0.1)
    b = torch.randn(num_modes)
    omega = torch.sqrt(k[:, 0]**2 + k[:, 1]**2 + k[:, 2]**2)
    
    shape = t.shape
    t_f, x_f, y_f, z_f = t.reshape(-1), x.reshape(-1), y.reshape(-1), z.reshape(-1)

    s = (omega.unsqueeze(1) * t_f.unsqueeze(0) +
            k[:, 0].unsqueeze(1) * x_f.unsqueeze(0) +
            k[:, 1].unsqueeze(1) * y_f.unsqueeze(0) +
            k[:, 2].unsqueeze(1) * z_f.unsqueeze(0) +
            b.unsqueeze(1))

    s_minus_c = s - 0.3
    s_minus_c_sq = s_minus_c ** 2
    f_double_prime = -0.2 * torch.exp(-10.0 * s_minus_c_sq) * (1.0 - 20.0 * s_minus_c_sq)

    F =  f_double_prime

    k1, k2, k3 = k[:, 0:1], k[:, 1:2], k[:, 2:3]
    w = omega.unsqueeze(1)

    Ex = torch.sum((k1 * k3 - k2 * w) * F, dim=0).view(shape)
    Ey = torch.sum((k2 * k3 + k1 * w) * F, dim=0).view(shape)
    Ez = torch.sum((k3**2 - w**2) * F, dim=0).view(shape)

    Bx = torch.sum((w * k2 + k1 * k3) * F, dim=0).view(shape)
    By = torch.sum((-w * k1 + k2 * k3) * F, dim=0).view(shape)
    Bz = torch.sum((-k1**2 - k2**2) * F, dim=0).view(shape)

    return torch.vstack((Ex, Ey, Ez, Bx, By, Bz))
