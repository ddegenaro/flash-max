import torch

c = 1.0

def f_gaussian(s):
    """The specified 1D function f(s)."""
    return 0.03 * torch.exp(-100.0 * (s - 0.3)**2)

def u(t_val, x_val, y_val, z_val):
    """
    Maps a 1D function f to the Maxwell solutions E and B
    using the specified second-order differential operators.
    """
    # Ensure coordinates require gradients for PyTorch autograd
    t = t_val.clone().detach().requires_grad_(True)
    x = x_val.clone().detach().requires_grad_(True)
    y = y_val.clone().detach().requires_grad_(True)
    z = z_val.clone().detach().requires_grad_(True)

    # Define u(t, x, y, z)
    r = torch.sqrt(x**2 + y**2 + z**2)
    u = f_gaussian(r - t) / (1e-8 + r)

    # ---------------------------------------------------------
    # 1. Compute First Derivatives
    # ---------------------------------------------------------
    grad_ones = torch.ones_like(u)
    du_dx, du_dy, du_dz, du_dt = torch.autograd.grad(
        outputs=u, inputs=(x, y, z, t),
        grad_outputs=grad_ones, create_graph=True
    )

    # ---------------------------------------------------------
    # 2. Compute Second Derivatives to construct E and B
    # ---------------------------------------------------------
    # E_x = \partial_x \partial_z u - \partial_y \partial_t u
    d2u_dxdz = torch.autograd.grad(du_dz, x, grad_outputs=grad_ones, retain_graph=True)[0]
    d2u_dydt = torch.autograd.grad(du_dt, y, grad_outputs=grad_ones, retain_graph=True)[0]
    E_x = d2u_dxdz - d2u_dydt

    # E_y = \partial_y \partial_z u + \partial_x \partial_t u
    d2u_dydz = torch.autograd.grad(du_dz, y, grad_outputs=grad_ones, retain_graph=True)[0]
    d2u_dxdt = torch.autograd.grad(du_dt, x, grad_outputs=grad_ones, retain_graph=True)[0]
    E_y = d2u_dydz + d2u_dxdt

    # E_z = \partial_z^2 u - \partial_t^2 u
    d2u_dz2 = torch.autograd.grad(du_dz, z, grad_outputs=grad_ones, retain_graph=True)[0]
    d2u_dt2 = torch.autograd.grad(du_dt, t, grad_outputs=grad_ones, retain_graph=True)[0]
    E_z = d2u_dz2 - d2u_dt2

    # B_x = \partial_t \partial_y u + \partial_x \partial_z u
    d2u_dtdy = torch.autograd.grad(du_dy, t, grad_outputs=grad_ones, retain_graph=True)[0]
    B_x = d2u_dtdy + d2u_dxdz # d2u_dxdz already computed

    # B_y = -\partial_t \partial_x u + \partial_y \partial_z u
    d2u_dtdx = torch.autograd.grad(du_dx, t, grad_outputs=grad_ones, retain_graph=True)[0]
    B_y = -d2u_dtdx + d2u_dydz # d2u_dydz already computed

    # B_z = -\partial_x^2 u - \partial_y^2 u
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=grad_ones, retain_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=grad_ones, retain_graph=True)[0]
    B_z = -d2u_dx2 - d2u_dy2

    return torch.vstack((E_x, E_y, E_z, B_x, B_y, B_z))