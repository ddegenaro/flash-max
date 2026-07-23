import torch
from torch import Tensor

c = 1.0

# Plane waves

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:

    return torch.vstack((
        torch.zeros(t.size(), device=t.device),
        torch.cos(y + z),
        -torch.cos(y + z),
        torch.zeros(t.size(), device=t.device),
        torch.zeros(t.size(), device=t.device),
        torch.zeros(t.size(), device=t.device),
    ))
