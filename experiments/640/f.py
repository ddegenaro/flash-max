import torch

def create_maxwell_solver(f_double_prime, num_components=100, seed=42):
    """
    Constructs a higher-order function mapping a 1D function's second derivative
    to exact (E, B) solutions of Maxwell's equations.
    """
    torch.manual_seed(seed)

    # Sample z_{ji} from N(0, 0.1).
    # Standard notation N(mean, variance) implies standard deviation is sqrt(0.1)
    Z = torch.randn(size=(num_components, 3)) * torch.sqrt(torch.tensor(0.1))
    z1, z2, z3 = Z[:, 0], Z[:, 1], Z[:, 2]

    # Sample b_j from N(0, 1)
    b = torch.randn(size=(num_components,))

    # Calculate angular frequency omega_j
    omega = torch.sqrt(z1**2 + z2**2 + z3**2)

    # Precompute the amplitude coefficients for E and B
    # based on the cross-derivative formulas provided.
    c_Ex = z1 * z3 - z2 * omega
    c_Ey = z2 * z3 + z1 * omega
    c_Ez = z3**2 - omega**2

    c_Bx = omega * z2 + z1 * z3
    c_By = -omega * z1 + z2 * z3
    c_Bz = -z1**2 - z2**2

    def compute_EB(t, x, y, z):
        """
        Evaluates the Electric and Magnetic fields at given coordinates.
        Supports scalars or N-dimensional numpy meshgrids.
        """
        x, y, z = torch.asarray(x), torch.asarray(y), torch.asarray(z)

        # Reshape parameters to broadcast across the spatial grid dimensions
        shape_expansion = (num_components,) + (1,) * x.ndim

        w_exp = omega.reshape(shape_expansion).to(t.device)
        z1_exp = z1.reshape(shape_expansion).to(t.device)
        z2_exp = z2.reshape(shape_expansion).to(t.device)
        z3_exp = z3.reshape(shape_expansion).to(t.device)
        b_exp = b.reshape(shape_expansion).to(t.device)

        # Calculate the phase argument S_j for all components and spatial points
        S = w_exp * t + z1_exp * x + z2_exp * y + z3_exp * z + b_exp

        # Evaluate the 1D function's second derivative
        f_ddot_vals = f_double_prime(S)

        # Compute fields by summing over the 100 components (axis=0)
        # The 1e-2 factor scales the sum as defined in u(t,x,y,z)
        Ex = 1e-2 * torch.sum(f_ddot_vals * c_Ex.reshape(shape_expansion), axis=0)
        Ey = 1e-2 * torch.sum(f_ddot_vals * c_Ey.reshape(shape_expansion), axis=0)
        Ez = 1e-2 * torch.sum(f_ddot_vals * c_Ez.reshape(shape_expansion), axis=0)

        Bx = 1e-2 * torch.sum(f_ddot_vals * c_Bx.reshape(shape_expansion), axis=0)
        By = 1e-2 * torch.sum(f_ddot_vals * c_By.reshape(shape_expansion), axis=0)
        Bz = 1e-2 * torch.sum(f_ddot_vals * c_Bz.reshape(shape_expansion), axis=0)

        return torch.vstack((Ex, Ey, Ez, Bx, By, Bz))

    return compute_EB

def f_double_prime(s):
    diff = s - 0.3
    return -20.0 * (1.0 - 200.0 * diff**2) * torch.exp(-100.0 * diff**2)

c = 1.0
u = create_maxwell_solver(f_double_prime=f_double_prime)