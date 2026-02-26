import torch
from torch import Tensor

# c MUST BE DEFINED
c: float = 1.

# u(t, x, [y, z]) MUST BE DEFINED
def f(s: Tensor) -> Tensor:
    return torch.exp(-5. * s**2) / 5.

# coordinate conversion
def r(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    return torch.sqrt(x**2 + y**2 + z**2)

# superposition of plane waves
# def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
#     return f(x + t) + f(y + t) + f(z + t)

# radial solution
# def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
#     R = r(x, y, z)
#     return (1 / R) * f(R - t)

sqrt2 = torch.sqrt(torch.tensor(2.))
sqrt3 = torch.sqrt(torch.tensor(3.))

# def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
#     return f(sqrt2 * t + x + y) + f(y + t) + f(z + t)
    
def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    return f(sqrt3 * t + x + y + z) + f(y + t) + f(z + t)