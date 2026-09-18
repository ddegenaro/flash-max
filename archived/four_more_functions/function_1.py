import torch
from torch import Tensor

c = 1.0

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
	r_sq = x**2+y**2+z**2

	E_x = ((y*(r_sq)**(3/2)+x*z*(r_sq-3))*torch.sin(-(r_sq)**(1/2)+t)-(3*x*z*(r_sq)**(1/2)+y*(r_sq))*torch.cos(-(r_sq)**(1/2)+t))/(r_sq)**(5/2)
	E_y =  -((x*(r_sq)**(3/2)-y*z*(r_sq-3))*torch.sin(-(r_sq)**(1/2)+t)-(-3*y*z*(r_sq)**(1/2)+x*(r_sq))*torch.cos(-(r_sq)**(1/2)+t))/(r_sq)**(5/2)
	E_z =  -((x**4+(2*y**2+z**2-1)*x**2+y**4+(z**2-1)*y**2+2*z**2)*torch.sin(-(r_sq)**(1/2)+t)-(r_sq)**(1/2)*torch.cos(-(r_sq)**(1/2)+t)*(x**2+y**2-2*z**2))/(r_sq)**(5/2)
	B_x =  ((-y*(r_sq)**(3/2)+x*z*(r_sq-3))*torch.sin(-(r_sq)**(1/2)+t)+(-3*x*z*(r_sq)**(1/2)+y*(r_sq))*torch.cos(-(r_sq)**(1/2)+t))/(r_sq)**(5/2)
	B_y =  1/(r_sq)**(5/2)*((x*(r_sq)**(3/2)+y*z*(r_sq-3))*torch.sin(-(r_sq)**(1/2)+t)-(3*y*z*(r_sq)**(1/2)+x*(r_sq))*torch.cos(-(r_sq)**(1/2)+t))
	B_z =  -((x**4+(2*y**2+z**2-1)*x**2+y**4+(z**2-1)*y**2+2*z**2)*torch.sin(-(r_sq)**(1/2)+t)-(r_sq)**(1/2)*torch.cos(-(r_sq)**(1/2)+t)*(x**2+y**2-2*z**2))/(r_sq)**(5/2)

	return torch.vstack((E_x, E_y, E_z, B_x, B_y, B_z))