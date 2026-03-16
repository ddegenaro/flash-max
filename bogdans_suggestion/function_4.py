import torch
from torch import Tensor

c = 1.0

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
	E_x = torch.zeros(t.size()).to(t.device)
	E_y = torch.zeros(t.size()).to(t.device)
	E_z = torch.zeros(t.size()).to(t.device)
	B_x = torch.zeros(t.size()).to(t.device)
	B_y = torch.zeros(t.size()).to(t.device)
	B_z = torch.zeros(t.size()).to(t.device)

	return torch.vstack((E_x, E_y, E_z, B_x, B_y, B_z))