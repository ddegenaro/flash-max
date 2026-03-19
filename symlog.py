import torch
from torch import Tensor

def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)