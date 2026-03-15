import torch
from torch import Tensor

c = 1.0

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:

	E_x = 0
	E_y =  0
	E_z =  0
	B_x =  0
	B_y =  0
	B_z =  0

	return torch.vstack((E_x, E_y, E_z, B_x, B_y, B_z))