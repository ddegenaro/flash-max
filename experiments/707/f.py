import torch
from torch import Tensor

c = 1.0
sqrt3 = torch.sqrt(torch.tensor(3))

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    cos_arg = torch.cos(sqrt3 * t + x + y + z)
    return torch.vstack((
        (sqrt3 - 1) * cos_arg,
        -(sqrt3 + 1) * cos_arg,
        2 * cos_arg,
        -(sqrt3 + 1) * cos_arg,
        (sqrt3 - 1) * cos_arg,
        2 * cos_arg
    ))