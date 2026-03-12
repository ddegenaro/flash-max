import torch
from torch import Tensor

c = 1.0

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
	r_sq = x**2+y**2+z**2

	E_x = 6/(r_sq)**(5/2)*(-1/6*y*(r_sq)**(3/2)*torch.cos(-(r_sq)**(1/2)+t)-1/6*y*(r_sq)*torch.sin(-(r_sq)**(1/2)+t)+1/3*t*y**3-1/3*x*y**2*z+1/3*t*(x**2+z**2)*y+z*x*(t**2-1/3*x**2-1/3*z**2))
	E_y =  6*(1/6*x*(r_sq)**(3/2)*torch.cos(-(r_sq)**(1/2)+t)+1/6*x*(r_sq)*torch.sin(-(r_sq)**(1/2)+t)-1/3*t*x**3-1/3*x**2*y*z-1/3*t*(y**2+z**2)*x+z*(t**2-1/3*y**2-1/3*z**2)*y)/(r_sq)**(5/2)
	E_z =  (-2*x**4+(-2*t**2-4*y**2-6*z**2)*x**2-2*y**4+(-2*t**2-6*z**2)*y**2+4*t**2*z**2-4*z**4)/(r_sq)**(5/2)
	B_x =  (-x*z*(r_sq-3)*(r_sq)**(1/2)*torch.cos(-(r_sq)**(1/2)+t)-3*x*z*(r_sq)*torch.sin(-(r_sq)**(1/2)+t)+3*(-1/3*x**3*z-4/3*t*x**2*y+z*(t**2-1/3*y**2-1/3*z**2)*x-4/3*t*y*(y**2+z**2))*(r_sq)**(1/2))/(r_sq)**3
	B_y =  (-y*z*(r_sq-3)*(r_sq)**(1/2)*torch.cos(-(r_sq)**(1/2)+t)-3*y*z*(r_sq)*torch.sin(-(r_sq)**(1/2)+t)+3*(-1/3*y**3*z+4/3*t*x*y**2+z*(t**2-1/3*x**2-1/3*z**2)*y+4/3*t*x*(x**2+z**2))*(r_sq)**(1/2))/(r_sq)**3
	B_z =  ((x**4+(2*y**2+z**2-1)*x**2+y**4+(z**2-1)*y**2+2*z**2)*(r_sq)**(1/2)*torch.cos(-(r_sq)**(1/2)+t)+(r_sq)*(x**2+y**2-2*z**2)*torch.sin(-(r_sq)**(1/2)+t)-(r_sq)**(1/2)*(x**4+(t**2+2*y**2+3*z**2)*x**2+y**4+(t**2+3*z**2)*y**2-2*t**2*z**2+2*z**4))/(r_sq)**3

	return torch.vstack((E_x, E_y, E_z, B_x, B_y, B_z))