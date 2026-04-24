from typing import Callable, Union

import torch

def random_data(
    num_samples: int = 1000,
    spatial_dim: int = 3,
    mins: list[float] = [0., 0., 0., 0.],
    maxes: list[float] = [1., 1., 1., 1.],
    f: Callable = None,
    noise_scale: float = 0,
    restrict_time: bool = False,
    add_bc: bool = False,
    do_masking: bool = True
) -> tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
    
    """
    Generate a random physical dataset.

    Args:
        num_samples (`int`): The number of samples to generate.
        dim (`int`): The number of spatial dimensions to involve.
        mins (`list[float]`): The minimum value each of the spatio-temporal dimensions should take.
            The first dimension is interpreted as time.
        maxes (`list[float]`): The corresponding maximum values.
        f (`Callable`): The function for generating targets. Expected signature is `f(t, x, [y, z])`.
        noise_scale (`float`): The scale of the noise to add to the targets.
        restrict_time (`bool`): Whether to restrict all time inputs only to the endpoints.
        add_bc (`bool`): Whether to use boundary conditions in the data. Cuts number of data points into
            five if 2D or seven if 3D, and makes each "piece" of the data correspond to samples
            enforcing the boundary conditions.
        
    Returns:
        `tuple[torch.Tensor, Union[torch.Tensor, None]]` where the first entry is the inputs,
            the second is the targets (`None` if `f` is `None`),
            and the third is the mask (`None` if `f` is None).
    """

    l_mins, l_maxes = len(mins), len(maxes)
    assert l_mins == l_maxes == 1 + spatial_dim, (
        f'spatial_dim should be one less than number of mins/maxes, but got: spatial_dim={spatial_dim}, {l_mins} mins, {l_maxes} maxes'
    )

    if not add_bc:
        if restrict_time:
            sample = torch.rand(
                (spatial_dim, round(num_samples)), dtype=torch.float32
            )
            
            ones = torch.ones(
                (1, round(num_samples)), dtype=torch.float32
            )
            ones[:, :round(num_samples / 2)] *= mins[0]
            ones[:, round(num_samples / 2):] *= maxes[0]
            
            sample = torch.cat((ones, sample))
            
            for i in range(1, 1 + spatial_dim):
                sample[i, :] *= (maxes[i] - mins[i])
                sample[i, :] += mins[i]
            
        else:
            sample = torch.rand(
                (1 + spatial_dim, round(num_samples)), dtype=torch.float32
            )

            for i in range(1 + spatial_dim):
                sample[i, :] *= (maxes[i] - mins[i])
                sample[i, :] += mins[i]
            
    if add_bc:
        
        if spatial_dim == 2:
            points_per_piece = round(num_samples / 5)
        elif spatial_dim == 3:
            points_per_piece = round(num_samples / 7)
        else:
            raise ValueError(f"Spatial dim was {spatial_dim} but should be 2 or 3.")
        
        sample = torch.rand(
            (spatial_dim + 1, round(num_samples)), dtype=torch.float32
        )
        
        for i in range(1 + spatial_dim):
            sample[i, :] *= (maxes[i] - mins[i])
            sample[i, :] += mins[i]
        
        # first piece, set t = t_min
        sample[0, 0*points_per_piece:1*points_per_piece] = mins[0]
        
        # second piece, set x to x_min
        sample[1, 1*points_per_piece:2*points_per_piece] = mins[1]
        
        # third piece, set x to x_max
        sample[1, 2*points_per_piece:3*points_per_piece] = maxes[1]
        
        # fourth piece, set y to y_min
        sample[2, 3*points_per_piece:4*points_per_piece] = mins[2]
        
        if spatial_dim == 2:
            # fifth piece, set y to y_max, go to end and use remainder here
            sample[2, 4*points_per_piece:] = maxes[2]
        else:
            # fifth piece, set y to y_max, stop appropriately for more pieces
            sample[2, 4*points_per_piece:5*points_per_piece] = maxes[2]
            
            # sixth piece, set z to z_min
            sample[3, 5*points_per_piece:6*points_per_piece] = mins[3]
            
            # seventh piece, set z to z_max
            sample[3, 6*points_per_piece:] = maxes[3]

    if f is not None:
        targets: torch.Tensor = f(*sample).transpose(0, -1)
        targets += torch.randn(*targets.shape) * noise_scale
        
        mask = torch.ones_like(targets)
        
        if add_bc and do_masking:
            
            # second-third pieces, E_x is ignored
            mask[1*points_per_piece:3*points_per_piece, 0] = 0
            
            # fourth-fifth pieces, E_y is ignored
            mask[3*points_per_piece:5*points_per_piece, 1] = 0
            
            try:
                # sixth-seventh pieces, E_z is ignored
                mask[5*points_per_piece:7*points_per_piece, 2] = 0
            except:
                pass
            
            # all pieces but first, B_x, B_y, B_z are ignored
            mask[1*points_per_piece:, 3:] = 0
            
        mask = mask.bool()
        
        return sample.transpose(0, -1), targets, mask
    else:
        return sample.transpose(0, -1), None, None
    
def grid_data(
    num_samples: int = 1000,
    spatial_dim: int = 3,
    mins: list[float] = [0., 0., 0., 0.],
    maxes: list[float] = [1., 1., 1., 1.],
    f: Callable = None,
    noise_scale: float = 0,
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
                steps=round(num_samples),
                dtype=torch.float32
            ) for _ in range(spatial_dim)]
        )

        ones = torch.ones(
            (1, round(num_samples)), dtype=torch.float32
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
                steps=round(num_samples),
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