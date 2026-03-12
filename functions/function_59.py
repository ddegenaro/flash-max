import torch
from torch import Tensor

c = 1.0

def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
	r_sq = x**2+y**2+z**2

	E_x = (-y*(r_sq)**2*torch.cos(-(r_sq)**(1/2)+t)-y*(r_sq)**(3/2)*torch.sin(-(r_sq)**(1/2)+t)+(r_sq)*(y**3+(x**2+z**2)*y-3*x*z)*torch.cos((r_sq)**(1/2)+t)+3*(1/3*(-y**3-x*y**2*z+(-x**2-z**2)*y-x*z*(x**2+z**2-3))*torch.sin((r_sq)**(1/2)+t)+x*z*(t**2-1/3*x**2-1/3*y**2-1/3*z**2))*(r_sq)**(1/2))/(r_sq)**3
	E_y =  (x*(r_sq)**2*torch.cos(-(r_sq)**(1/2)+t)+x*(r_sq)**(3/2)*torch.sin(-(r_sq)**(1/2)+t)-(r_sq)*(x**3+(y**2+z**2)*x+3*y*z)*torch.cos((r_sq)**(1/2)+t)+3*(1/3*(x**3-x**2*y*z+(y**2+z**2)*x-y*z*(y**2+z**2-3))*torch.sin((r_sq)**(1/2)+t)+y*z*(t**2-1/3*x**2-1/3*y**2-1/3*z**2))*(r_sq)**(1/2))/(r_sq)**3
	E_z =  ((x**4+(2*y**2+z**2-1)*x**2+y**4+(z**2-1)*y**2+2*z**2)*(r_sq)**(1/2)*torch.sin((r_sq)**(1/2)+t)+torch.cos((r_sq)**(1/2)+t)*(r_sq)*(x**2+y**2-2*z**2)-(r_sq)**(1/2)*(x**4+(t**2+2*y**2+3*z**2)*x**2+y**4+(t**2+3*z**2)*y**2-2*t**2*z**2+2*z**4))/(r_sq)**3
	B_x =  (-x*z*(r_sq-3)*(r_sq)**(1/2)*torch.cos(-(r_sq)**(1/2)+t)-3*x*z*(r_sq)*torch.sin(-(r_sq)**(1/2)+t)-(x**2*y+y**3+y*z**2-3*x*z)*(r_sq)*torch.sin((r_sq)**(1/2)+t)-2*(1/2*(x**3*z+x**2*y+z*(y**2+z**2-3)*x+y**3+y*z**2)*torch.cos((r_sq)**(1/2)+t)+t*y*(r_sq))*(r_sq)**(1/2))/(r_sq)**3
	B_y =  (-y*z*(r_sq-3)*(r_sq)**(1/2)*torch.cos(-(r_sq)**(1/2)+t)-3*y*z*(r_sq)*torch.sin(-(r_sq)**(1/2)+t)+(r_sq)*(x**3+x*y**2+x*z**2+3*y*z)*torch.sin((r_sq)**(1/2)+t)+2*(1/2*(-y**3*z+y**2*x-z*(x**2+z**2-3)*y+x**3+x*z**2)*torch.cos((r_sq)**(1/2)+t)+t*x*(r_sq))*(r_sq)**(1/2))/(r_sq)**3
	B_z =  ((x**4+(2*y**2+z**2-1)*x**2+y**4+(z**2-1)*y**2+2*z**2)*(r_sq)**(1/2)*torch.cos(-(r_sq)**(1/2)+t)+(r_sq)*(x**2+y**2-2*z**2)*torch.sin(-(r_sq)**(1/2)+t)+(x**4+(2*y**2+z**2-1)*x**2+y**4+(z**2-1)*y**2+2*z**2)*(r_sq)**(1/2)*torch.cos((r_sq)**(1/2)+t)-torch.sin((r_sq)**(1/2)+t)*(r_sq)*(x**2+y**2-2*z**2))/(r_sq)**3

	return torch.vstack((E_x, E_y, E_z, B_x, B_y, B_z))