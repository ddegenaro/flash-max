import torch
from sympy import *
import numpy as np

c = 1.0
c_sq = c**2

b1 = 1.0
b2 = 5.0
p = 2

t, x, y, z = symbols('t x y z')

r = sqrt(x**2 + y**2 + z**2)

s_minus = r - t
s_plus = r + t

u_sym = b1 * (s_minus**p) * exp(-b2 * s_minus**2) / r
v_sym = b1 * (s_minus**p) * exp(-b2 * s_minus**2) / r

dt_u = diff(u_sym, t)
dx_u = diff(u_sym, x)
dy_u = diff(u_sym, y)
dz_u = diff(u_sym, z)

dt_v = diff(v_sym, t)
dx_v = diff(v_sym, x)
dy_v = diff(v_sym, y)
dz_v = diff(v_sym, z)

dz_dz_u = diff(dz_u, z)
dt_dt_u = diff(dt_u, t)

dx_dz_u = diff(dz_u, x)
dy_dz_u = diff(dz_u, y)

dt_dy_u = diff(dy_u, t)
dt_dx_u = diff(dx_u, t)

dy_dt_v = diff(dt_v, y)
dx_dt_v = diff(dt_v, x)

dx_dz_v = diff(dz_v, x)
dy_dz_v = diff(dz_v, y)
dx_dx_v = diff(dx_v, x)
dy_dy_v = diff(dy_v, y)

# E from u
E_x_u = dx_dz_u
E_y_u = dy_dz_u
E_z_u = dz_dz_u - (dt_dt_u / c_sq)

# B from u
B_x_u = dt_dy_u / c_sq
B_y_u = -dt_dx_u / c_sq
B_z_u = 0

# E from v
E_x_v = -dy_dt_v
E_y_v = dx_dt_v
E_z_v = 0

# B from v
B_x_v = dx_dz_v
B_y_v = dy_dz_v
B_z_v = -dx_dx_v - dy_dy_v

args = [t, x, y, z]

E_x = lambdify(args, E_x_u + E_x_v, 'numpy')
E_y = lambdify(args, E_y_u + E_y_v, 'numpy')
E_z = lambdify(args, E_z_u + E_z_v, 'numpy')

B_x = lambdify(args, B_x_u + B_x_v, 'numpy')
B_y = lambdify(args, B_y_u + B_y_v, 'numpy')
B_z = lambdify(args, B_z_u + B_z_v, 'numpy')

def u(t, x, y, z):
    t_np = t.cpu().numpy().reshape(-1, 1)
    x_np = x.cpu().numpy().reshape(-1, 1)
    y_np = y.cpu().numpy().reshape(-1, 1)
    z_np = z.cpu().numpy().reshape(-1, 1)
    
    result = torch.hstack((
        E_x(t, x, y, z),
        E_y(t, x, y, z),
        E_z(t, x, y, z),
        B_x(t, x, y, z),
        B_y(t, x, y, z),
        B_z(t, x, y, z)
    ))
    
    return result

if __name__ == '__main__':
    bs = 32
    result = u(
        torch.ones((bs, 1)),
        torch.ones((bs, 1)),
        torch.ones((bs, 1)),
        torch.ones((bs, 1))
    )
    
    good = torch.Size([bs, 6])
    
    assert torch.any(result != 0), f'result is all 0'
    
    assert result.shape == good, f'shape is {result.shape} but should be {good}'
    
