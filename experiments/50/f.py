import torch

# c MUST BE DEFINED
c: float = 10.

# # u(t, x, [y, z]) MUST BE DEFINED
def u(t, x, y):
    return x**2 + y**2
    
def f(s):
    return torch.exp(-5. * s**2) / 5.

# sqrt2 = torch.sqrt(torch.tensor(2.))
# sqrt5 = torch.sqrt(torch.tensor(5.))

# def f(s: torch.Tensor) -> torch.Tensor:
#     return torch.exp(-(s - a) ** 2)

# def g(s: torch.Tensor) -> torch.Tensor:
#     return 3 * torch.exp(-4 * (s - b) ** 2)

# def u(
#     t: torch.Tensor,
#     x: torch.Tensor,
#     y: torch.Tensor
# ) -> torch.Tensor:
#     return 2 * c**2 * t**2 + x**2 + y**2
