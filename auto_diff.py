from typing import Callable

import torch
from torch import Tensor

FIELD_COMPONENTS = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']
COMPONENT_IDX = {name: i for i, name in enumerate(FIELD_COMPONENTS)}
INPUT_VARS = {'t': 0, 'x': 1, 'y': 2, 'z': 3}

def diff(
    u: Callable,
    component: str,
    first_var: str,
    second_var: str | None = None
) -> Callable:
    """
    Returns a callable that computes a first or second order partial derivative
    of a field component output by u.

    Args:
        u:            callable with signature u(t, x, y, z) -> (6, N) tensor
        component:    one of 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'
        first_var:    one of 't', 'x', 'y', 'z'
        second_var:   one of 't', 'x', 'y', 'z', or None for first-order

    Returns:
        A callable f(t, x, y, z) -> (1, N) tensor of the requested derivative.

    Examples:
        dEx_dx  = diff(u, 'Ex', 'x')        # dEx/dx
        d2Bz_yx = diff(u, 'Bz', 'y', 'x')   # d²Bz/dx dy
    """
    comp_idx = COMPONENT_IDX[component]
    vars_list = list(INPUT_VARS.keys())  # ['t', 'x', 'y', 'z']

    if first_var not in INPUT_VARS:
        raise ValueError(f"first_var must be one of {vars_list}, got '{first_var}'")
    if second_var is not None and second_var not in INPUT_VARS:
        raise ValueError(f"second_var must be one of {vars_list} or None, got '{second_var}'")

    def derivative(t: Tensor, x: Tensor, y: Tensor, z: Tensor):
        inputs = [t, x, y, z]

        # Enable gradients for all inputs
        for inp in inputs:
            inp.requires_grad_(True)

        # --- First derivative ---
        output = u(t, x, y, z)           # (6, N)
        scalar = output[comp_idx].sum()   # scalar for autograd

        first_idx = INPUT_VARS[first_var]
        grad_first, = torch.autograd.grad(
            outputs=scalar,
            inputs=inputs[first_idx],
            create_graph=second_var is not None,  # keep graph only if needed
            retain_graph=second_var is not None,
        )                                 # (1, N)

        if second_var is None:
            return grad_first

        # --- Second derivative ---
        scalar2 = grad_first.sum()
        second_idx = INPUT_VARS[second_var]
        grad_second, = torch.autograd.grad(
            outputs=scalar2,
            inputs=inputs[second_idx],
            create_graph=False,
        )                                 # (1, N)

        return grad_second

    return derivative