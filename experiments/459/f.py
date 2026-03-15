import torch
from torch import Tensor

c = 1.0

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
	r_sq = x**2+y**2+z**2

	E_x = -1/(r_sq)**(5/2)*((x**3*z+x**2*y+z*(y**2+z**2-3)*x+y**3+y*z**2)*torch.cos(-(r_sq)**(1/2)+t)-(r_sq)**(1/2)*torch.sin(-(r_sq)**(1/2)+t)*(x**2*y+y**3+y*z**2-3*x*z))
	E_y =  -((-x**3+x**2*y*z+(-y**2-z**2)*x+y*z*(y**2+z**2-3))*torch.cos(-(r_sq)**(1/2)+t)+torch.sin(-(r_sq)**(1/2)+t)*(x**3+(y**2+z**2)*x+3*y*z)*(r_sq)**(1/2))/(r_sq)**(5/2)
	E_z =  ((x**4+(2*y**2+z**2-1)*x**2+y**4+(z**2-1)*y**2+2*z**2)*torch.cos(-(r_sq)**(1/2)+t)+(r_sq)**(1/2)*torch.sin(-(r_sq)**(1/2)+t)*(x**2+y**2-2*z**2))/(r_sq)**(5/2)
	B_x =  ((x**3*z+x**2*y+z*(y**2+z**2-3)*x+y**3+y*z**2)*torch.sin(-(r_sq)**(1/2)+t)+(r_sq)**(1/2)*torch.cos(-(r_sq)**(1/2)+t)*(x**2*y+y**3+y*z**2-3*x*z))/(r_sq)**(5/2)
	B_y =  -((x**3-x**2*y*z+(y**2+z**2)*x-y*z*(y**2+z**2-3))*torch.sin(-(r_sq)**(1/2)+t)+(x**3+(y**2+z**2)*x+3*y*z)*torch.cos(-(r_sq)**(1/2)+t)*(r_sq)**(1/2))/(r_sq)**(5/2)
	B_z =  -((x**4+(2*y**2+z**2-1)*x**2+y**4+(z**2-1)*y**2+2*z**2)*torch.sin(-(r_sq)**(1/2)+t)-(r_sq)**(1/2)*torch.cos(-(r_sq)**(1/2)+t)*(x**2+y**2-2*z**2))/(r_sq)**(5/2)

	return torch.vstack((E_x, E_y, E_z, B_x, B_y, B_z))