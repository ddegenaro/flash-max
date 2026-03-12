import torch
from torch import Tensor

c = 1.0

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
	r_sq = x**2+y**2+z**2

	E_x = (x*z*(r_sq-3)*(r_sq)**(1/2)*torch.sin(-(r_sq)**(1/2)+t)-3*x*z*(r_sq)*torch.cos(-(r_sq)**(1/2)+t)-x*z*(r_sq-3)*(r_sq)**(1/2)*torch.sin((r_sq)**(1/2)+t)+4*(r_sq)*(t*y*(r_sq)**(1/2)-3/4*torch.cos((r_sq)**(1/2)+t)*x*z))/(r_sq)**3
	E_y =  (y*z*(r_sq-3)*(r_sq)**(1/2)*torch.sin(-(r_sq)**(1/2)+t)-3*y*z*(r_sq)*torch.cos(-(r_sq)**(1/2)+t)-y*z*(r_sq-3)*(r_sq)**(1/2)*torch.sin((r_sq)**(1/2)+t)-4*(t*x*(r_sq)**(1/2)+3/4*torch.cos((r_sq)**(1/2)+t)*y*z)*(r_sq))/(r_sq)**3
	E_z =  (-(x**4+(2*y**2+z**2-1)*x**2+y**4+(z**2-1)*y**2+2*z**2)*(r_sq)**(1/2)*torch.sin(-(r_sq)**(1/2)+t)+(r_sq)*(x**2+y**2-2*z**2)*torch.cos(-(r_sq)**(1/2)+t)+(x**4+(2*y**2+z**2-1)*x**2+y**4+(z**2-1)*y**2+2*z**2)*(r_sq)**(1/2)*torch.sin((r_sq)**(1/2)+t)+torch.cos((r_sq)**(1/2)+t)*(r_sq)*(x**2+y**2-2*z**2))/(r_sq)**3
	B_x =  6*(1/6*y*(r_sq)*torch.cos(-(r_sq)**(1/2)+t)-1/6*y*(r_sq)**(3/2)*torch.sin(-(r_sq)**(1/2)+t)-1/6*y*(r_sq)*torch.cos((r_sq)**(1/2)+t)-1/6*y*(r_sq)**(3/2)*torch.sin((r_sq)**(1/2)+t)+x*z*(t**2-1/3*x**2-1/3*y**2-1/3*z**2))/(r_sq)**(5/2)
	B_y =  6*(-1/6*x*(r_sq)*torch.cos(-(r_sq)**(1/2)+t)+1/6*x*(r_sq)**(3/2)*torch.sin(-(r_sq)**(1/2)+t)+1/6*x*(r_sq)*torch.cos((r_sq)**(1/2)+t)+1/6*x*(r_sq)**(3/2)*torch.sin((r_sq)**(1/2)+t)+y*z*(t**2-1/3*x**2-1/3*y**2-1/3*z**2))/(r_sq)**(5/2)
	B_z =  (-2*x**4+(-2*t**2-4*y**2-6*z**2)*x**2-2*y**4+(-2*t**2-6*z**2)*y**2+4*t**2*z**2-4*z**4)/(r_sq)**(5/2)

	return torch.vstack((E_x, E_y, E_z, B_x, B_y, B_z))