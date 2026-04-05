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
    # Wrap inputs as new leaf tensors that require grad.
    # Do NOT detach — just enable grad on fresh tensors with the same data.
    t = t_val.detach().requires_grad_(True)
    x = x_val.detach().requires_grad_(True)
    y = y_val.detach().requires_grad_(True)
    z = z_val.detach().requires_grad_(True)

    # Define u(t, x, y, z)
    # Use epsilon inside sqrt to avoid zero-gradient / NaN at the origin
    eps = 1e-8
    r = torch.sqrt(x**2 + y**2 + z**2 + eps)
    u_val = f_gaussian(r - t) / r          # r already has eps, no second guard needed

    # ---------------------------------------------------------
    # 1. Compute First Derivatives
    # ---------------------------------------------------------
    grad_ones = torch.ones_like(u_val)
    du_dx, du_dy, du_dz, du_dt = torch.autograd.grad(
        outputs=u_val, inputs=(x, y, z, t),
        grad_outputs=grad_ones, create_graph=True
    )

    # ---------------------------------------------------------
    # 2. Compute Second Derivatives to construct E and B
    # ---------------------------------------------------------
    grad_ones = torch.ones_like(du_dx)   # same shape, reuse name

    # E_x = ∂_x ∂_z u  −  ∂_y ∂_t u
    d2u_dxdz = torch.autograd.grad(du_dz, x, grad_outputs=torch.ones_like(du_dz), retain_graph=True)[0]
    d2u_dydt = torch.autograd.grad(du_dt, y, grad_outputs=torch.ones_like(du_dt), retain_graph=True)[0]
    E_x = d2u_dxdz - d2u_dydt

    # E_y = ∂_y ∂_z u  +  ∂_x ∂_t u
    d2u_dydz = torch.autograd.grad(du_dz, y, grad_outputs=torch.ones_like(du_dz), retain_graph=True)[0]
    d2u_dxdt = torch.autograd.grad(du_dt, x, grad_outputs=torch.ones_like(du_dt), retain_graph=True)[0]
    E_y = d2u_dydz + d2u_dxdt

    # E_z = ∂_z² u  −  ∂_t² u
    d2u_dz2 = torch.autograd.grad(du_dz, z, grad_outputs=torch.ones_like(du_dz), retain_graph=True)[0]
    d2u_dt2 = torch.autograd.grad(du_dt, t, grad_outputs=torch.ones_like(du_dt), retain_graph=True)[0]
    E_z = d2u_dz2 - d2u_dt2

    # B_x = ∂_t ∂_y u  +  ∂_x ∂_z u
    d2u_dtdy = torch.autograd.grad(du_dy, t, grad_outputs=torch.ones_like(du_dy), retain_graph=True)[0]
    B_x = d2u_dtdy + d2u_dxdz

    # B_y = −∂_t ∂_x u  +  ∂_y ∂_z u
    d2u_dtdx = torch.autograd.grad(du_dx, t, grad_outputs=torch.ones_like(du_dx), retain_graph=True)[0]
    B_y = -d2u_dtdx + d2u_dydz

    # B_z = −∂_x² u  −  ∂_y² u
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), retain_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy), retain_graph=True)[0]
    B_z = -d2u_dx2 - d2u_dy2

    return torch.vstack((E_x, E_y, E_z, B_x, B_y, B_z))