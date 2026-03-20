import torch

v = 1.0
a = 1.0
c = 1.0

def bump_function(xi, a=1.0):
    """
    1D compactly supported bump function.
    Strictly zero for |xi| >= a.
    """
    # Clamp to avoid division by zero in the exponent during autograd
    safe_xi = torch.clamp(xi, -a + 1e-6, a - 1e-6)
    val = torch.exp(-1.0 / (1.0 - (safe_xi / a)**2))

    # Enforce strict zero boundary (compact support)
    mask = torch.abs(xi) < a
    return torch.where(mask, val, torch.zeros_like(xi))

@torch.set_grad_enabled(True)
def u(t, x, y, z):
    """
    Computes E and B fields for the compactly supported EMP using Autograd.
    """
    # Ensure our coordinates require gradients for autodiff
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    # 1. Compute Scalar Wave Phi
    r = 5*torch.sqrt(x**2 + y**2 + z**2 + 1e-8) # Add epsilon to avoid singularity at origin
    f_out = 0.01 * bump_function(r - 5*v * (t + 0.3), a)
    f_in = 0.01 * bump_function(-r - 5*v * (t + 0.3), a)
    Phi = (f_out - f_in) / r

    # 2. Define Vector Potential A = curl(Phi * z_hat)
    # A = [dPhi/dy, -dPhi/dx, 0]
    grad_outputs = torch.ones_like(Phi)

    # We use create_graph=True because we need to take derivatives OF these derivatives later
    # breakpoint()
    dPhi_dx, dPhi_dy = torch.autograd.grad(
        outputs=Phi, inputs=(x, y),
        grad_outputs=grad_outputs, create_graph=True
    )

    A_x = dPhi_dy
    A_y = -dPhi_dx
    # A_z is conceptually 0

    # 3. Compute Electric Field E = -dA/dt
    E_x = -torch.autograd.grad(A_x, t, grad_outputs=grad_outputs, create_graph=True)[0]
    E_y = -torch.autograd.grad(A_y, t, grad_outputs=grad_outputs, create_graph=True)[0]
    E_z = torch.zeros_like(E_x) # Since A_z is 0 and A_x, A_y don't depend on z in a way that generates E_z here

    # 4. Compute Magnetic Field B = curl(A)
    # B = [-dA_y/dz, dA_x/dz, dA_y/dx - dA_x/dy]
    dA_y_dz = torch.autograd.grad(A_y, z, grad_outputs=grad_outputs, create_graph=True)[0]
    dA_x_dz = torch.autograd.grad(A_x, z, grad_outputs=grad_outputs, create_graph=True)[0]
    dA_y_dx = torch.autograd.grad(A_y, x, grad_outputs=grad_outputs, create_graph=True)[0]
    dA_x_dy = torch.autograd.grad(A_x, y, grad_outputs=grad_outputs, create_graph=True)[0]

    B_x = -dA_y_dz
    B_y = dA_x_dz
    B_z = dA_y_dx - dA_x_dy

    return torch.vstack((E_x, E_y, E_z, B_x, B_y, B_z)).detach()
