import torch
from torch import Tensor

c = 1.0

# New radial waves

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    r = torch.sqrt(x**2 + y**2 + z**2)
    rho = torch.sqrt(x**2 + y**2)

    f1 = (t + 1 - r) / (r * rho)
    
    f1_r = f1 / r

    return torch.vstack((
        f1_r * z * x,
        f1_r * z * y,
        f1_r * -rho**2,
        f1 * -y,
        f1 * x,
        torch.zeros(t.size())
    ))
