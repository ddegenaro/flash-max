from typing import Callable, Union

import torch

def random_data(
    num_samples: int = 100,
    spatial_dim: int = 2,
    mins: list[float] = [0., 0., 0.],
    maxes: list[float] = [1., 1., 1.],
    f: Callable = None,
    noise_scale: float = 1e-3,
    restrict_time: bool = False
) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
    
    """
    Generate a random physical dataset.

    Args:
        num_samples (`int`): The number of samples to generate.
        dim (`int`): The number of spatial dimensions to involve.
        mins (`list[float]`): The minimum value each of the spatio-temporal dimensions should take. The first dimension is interpreted as time.
        maxes (`list[float]`): The corresponding maximum values.
        f (`Callable`): The function for generating targets. Expected signature is `f(t, x, [y, z])`.
        noise_scale (`float`): The scale of the noise to add to the targets.
        restrict_time (`bool`): Whether to restrict all time inputs only to the endpoints.
        
    Returns:
        `tuple[torch.Tensor, Union[torch.Tensor, None]]` where the first entry is the inputs and the second is the targets (`None` if `f` is `None`).
    """

    l_mins, l_maxes = len(mins), len(maxes)
    assert l_mins == l_maxes == 1 + spatial_dim, (
        f'spatial_dim should be one less than number of mins/maxes, but got: spatial_dim={spatial_dim}, {l_mins} mins, {l_maxes} maxes'
    )

    if restrict_time:
        sample = torch.rand(
            (spatial_dim, num_samples), dtype=torch.float32
        )
        
        ones = torch.ones(
            (1, num_samples), dtype=torch.float32
        )
        ones[:, :round(num_samples / 2)] *= mins[0]
        ones[:, round(num_samples / 2):] *= maxes[0]
        
        sample = torch.cat((ones, sample))
        
        for i in range(1, 1 + spatial_dim):
            sample[i, :] *= (maxes[i] - mins[i])
            sample[i, :] += mins[i]
        
    else:
        sample = torch.rand(
            (1 + spatial_dim, num_samples), dtype=torch.float32
        )

        for i in range(1 + spatial_dim):
            sample[i, :] *= (maxes[i] - mins[i])
            sample[i, :] += mins[i]

    if f is not None:
        targets: torch.Tensor = f(*sample).transpose(0, -1)
        targets += torch.randn(*targets.shape) * noise_scale
        return sample.transpose(0, -1), targets
    else:
        return sample.transpose(0, -1), None
    
def grid_data(
    num_samples: int = 100,
    spatial_dim: int = 2,
    mins: list[float] = [0., 0., 0.],
    maxes: list[float] = [1., 1., 1.],
    f: Callable = None,
    noise_scale: float = 1e-3,
    restrict_time: bool = False
):
    """
    Generate a linspaced physical dataset.

    Args:
        num_samples (`int`): The number of samples to generate.
        dim (`int`): The number of spatial dimensions to involve.
        mins (`list[float]`): The minimum value each of the spatio-temporal dimensions should take. The first dimension is interpreted as time.
        maxes (`list[float]`): The corresponding maximum values.
        f (`Callable`): The function for generating targets. Expected signature is `f(t, x, [y, z])`.
        noise_scale (`float`): The scale of the noise to add to the targets.
        restrict_time (`bool`): Whether to restrict all time inputs only to the endpoints.
        
    Returns:
        `tuple[torch.Tensor, Union[torch.Tensor, None]]` where the first entry is the inputs and the second is the targets (`None` if `f` is `None`).
    """

    l_mins, l_maxes = len(mins), len(maxes)
    assert l_mins == l_maxes == 1 + spatial_dim, (
        f'spatial_dim should be one less than number of mins/maxes, but got: spatial_dim={spatial_dim}, {l_mins} mins, {l_maxes} maxes'
    )

    if restrict_time:
        sample = torch.vstack(
            [torch.linspace(
                start=0,
                end=1,
                steps=num_samples,
                dtype=torch.float32
            ) for _ in range(spatial_dim)]
        )

        ones = torch.ones(
            (1, num_samples), dtype=torch.float32
        )
        ones[:, :round(num_samples / 2)] *= mins[0]
        ones[:, round(num_samples / 2):] *= maxes[0]

        sample = torch.cat((ones, sample))

        for i in range(1, 1 + spatial_dim):
            sample[i, :] *= (maxes[i] - mins[i])
            sample[i, :] += mins[i]
    else:
        sample = torch.vstack(
            [torch.linspace(
                start=0,
                end=1,
                steps=num_samples,
                dtype=torch.float32
            ) for i in range(1 + spatial_dim)]
        )
        for i in range(1 + spatial_dim):
            sample[i, :] *= (maxes[i] - mins[i])
            sample[i, :] += mins[i]

    if f is not None:
        targets: torch.Tensor = f(*sample).transpose(0, -1)
        targets += torch.randn(*targets.shape) * noise_scale
        return sample.transpose(0, -1), targets
    else:
        return sample.transpose(0, -1), None