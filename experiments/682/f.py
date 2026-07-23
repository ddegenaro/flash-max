import torch

c = 1.0

def f_gaussian(s):
    return 0.0003 * torch.exp(-100.0 * (s - 0.3)**2)

def u(t, x, y, z):
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    r = torch.sqrt(x**2 + y**2 + z**2)
    u_val = f_gaussian(r - t) / (1e-8 + r)

    ones = torch.ones_like(u_val)

    # First derivatives — keep graph alive for all of them
    du_dx, du_dy, du_dz, du_dt = torch.autograd.grad(
        u_val, (x, y, z, t),
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True   # <-- explicit, even though create_graph implies it
    )

    def grad1(output, *inputs):
        """Compute one or more partials of `output` w.r.t. `inputs` cleanly."""
        return torch.autograd.grad(
            output, inputs,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,   # always retain — we may reuse the graph
            create_graph=False
        )

    # Second derivatives — each call is isolated and explicit
    (d2u_dzdx,) = grad1(du_dz, x)
    (d2u_dzdy,) = grad1(du_dz, y)
    (d2u_dzdz,) = grad1(du_dz, z)

    (d2u_dtdy,) = grad1(du_dt, y)
    (d2u_dtdx,) = grad1(du_dt, x)
    (d2u_dtdt,) = grad1(du_dt, t)

    (d2u_dydx,) = grad1(du_dy, x)  # for B_x cross-check if needed
    (d2u_dydt,) = grad1(du_dy, t)

    (d2u_dxdt,) = grad1(du_dx, t)
    (d2u_dxdx,) = grad1(du_dx, x)

    (d2u_dydy,) = grad1(du_dy, y)

    # Field components
    E_x = d2u_dzdx - d2u_dtdy
    E_y = d2u_dzdy + d2u_dtdx
    E_z = d2u_dzdz - d2u_dtdt

    B_x = d2u_dydt + d2u_dzdx
    B_y = -d2u_dxdt + d2u_dzdy
    B_z = -d2u_dxdx - d2u_dydy

    return torch.vstack((E_x, E_y, E_z, B_x, B_y, B_z)).detach()