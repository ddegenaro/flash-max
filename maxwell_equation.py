from typing import Callable

import torch
from torch import nn

from utils import PCNN, tensor_round
from data_sampler import grid_data

class MaxwellSimple(PCNN):

    def __init__(
        self,
        width: int = 10,
        c: float = 1.,
        input_dim: int = 1,
        output_dim: int = -1,
        activation: str = 'relu',
        do: float = 0.1,
        init = 'kaiming',
        keys = ('1+', '1-', '2+', '2-'),
        gain = None
    ):
        
        super().__init__(
            width=width,
            c=c,
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            do=do,
            init=init,
            gain=gain
        )
        
        self.Z_x = nn.ParameterDict() # Z_1, Z_2, Z_3
        self.W = nn.ParameterDict()
        self.b = nn.ParameterDict()
        
        self.keys = keys
        
        # breakpoint()
        
        for key in self.keys:
            self.Z_x[key] = nn.Parameter(torch.zeros(3, self.width))
            self.W[key] = nn.Parameter(torch.zeros(1, self.width))
            self.b[key] = nn.Parameter(torch.zeros(1, self.width))
            
            if self.init == 'kaiming normal':
                nn.init.kaiming_normal_(self.Z_x[key].data)
                nn.init.kaiming_normal_(self.W[key].data.T)
            elif self.init == 'xavier normal':
                nn.init.xavier_normal_(self.Z_x[key].data, gain=self.gain)
                nn.init.xavier_normal_(self.W[key].data.T, gain=self.gain)
            elif self.init == 'kaiming uniform':
                nn.init.kaiming_uniform_(self.Z_x[key].data)
                nn.init.kaiming_uniform_(self.W[key].data.T)
            elif self.init == 'xavier uniform':
                nn.init.xavier_uniform_(self.Z_x[key].data, gain=self.gain)
                nn.init.xavier_uniform_(self.W[key].data.T, gain=self.gain)
        
    def forward(self, t, x):
        
        X = torch.hstack((t, x))
        
        R = 0.
        
        for key in self.keys:
            
            if '+' in key:
                Z = torch.vstack((
                    torch.sqrt((self.Z_x[key] ** 2).sum(0, keepdim=True) + 1e-10), # Z_0
                    self.Z_x[key] # Z_1, Z_2, Z_3
                ))
            elif '-' in key:
                Z = torch.vstack((
                    -torch.sqrt((self.Z_x[key] ** 2).sum(0, keepdim=True) + 1e-10), # Z_0
                    self.Z_x[key] # Z_1, Z_2, Z_3
                ))
        
            A = self.dropout(self.activation(X @ Z + self.b[key])) * self.W[key]

            if '1' in key:
                P = torch.vstack((
                    -Z[1] * Z[3],
                    -Z[2] * Z[3],
                    Z[0]**2 - Z[3]**2,
                    -Z[0] * Z[2],
                    Z[0] * Z[1],
                    torch.zeros(self.width, device=Z.device)
                )).T
            elif '2' in key:
                P = torch.vstack((
                    Z[1] * Z[2],
                    -Z[0]**2 + Z[2]**2,
                    Z[2] * Z[3],
                    -Z[0] * Z[3],
                    torch.zeros(self.width, device=Z.device),
                    Z[0] * Z[1]
                )).T
                
            R += A @ P
        
        return R / self.width



class Maxwell(PCNN):

    def __init__(
        self,
        width: int = 10,
        c: float = 1.,
        input_dim: int = 1,
        output_dim: int = -1,
        activation: str = 'relu',
        # dropout_val: float = 0.1,
        init = 'kaiming',
        p_array: list[list[Callable]] = [[]],
        z_array: list[list[Callable]] = [[]]
    ):
        
        assert len(p_array) == len(z_array), f'Found {len(p_array)} p vectors, but {len(z_array)} z vectors.'
        
        if output_dim == -1: # interpret dimension automatically
            output_dim = len(p_array[0])
        
        for i, row in enumerate(p_array):
            assert len(row) == output_dim, f'p_array: Row {i} has {len(row)} callables, but output_dim is {output_dim}.'
            for j, func in enumerate(row):
                assert isinstance(func, Callable), f'p_array: Row {i}, col {j} is not Callable.'
            
        for i, row in enumerate(z_array):
            assert len(row) == len(z_array[0]), f'z_array: Row {i} has {len(row)} callables, but row 0 has {len(z_array[0])}.'
            for j, func in enumerate(row):
                assert isinstance(func, Callable), f'z_array: Row {i}, col {j} is not Callable.'
        
        super().__init__(
            width=width,
            c=c,
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            # dropout_val=dropout_val
            init=init
        )
        
        self.p_array = p_array # shape is N rows, 6 columns
        self.z_array = z_array # shape is N rows, M columns
        self.N = len(self.p_array) # N: number of p vectors, arbitrary
        self.M = len(self.z_array[0]) # M: number of z expressions per p, also arbitrary
        
        self.Z_x = nn.Parameter(torch.randn((self.input_dim, self.width, self.N, self.M)))
        
        self.b = nn.Parameter(torch.zeros((1, self.width, self.N, self.M)))

        self.W = nn.Parameter(torch.randn((self.width, 1, self.N, self.M)))
        
        nn.init.normal_(self.Z_x.data, 0, 1)
        nn.init.normal_(self.W.data, 0, 1)
        
    def forward(self, t, x):
        
        batch_size = t.shape[0]
        
        # Z_t (Z_0) is implicitly defined in terms of Z_x (Z_1,2,3)
        Z_t = torch.zeros(
            1, self.width, self.N, self.M,
            dtype=self.Z_x.dtype,
            device=self.Z_x.device
        )

        # NOTE: may be able to vectorize in special cases
        for j in range(self.N):
            for k in range(self.M):
                Z_t[0, :, j, k] = self.z_array[j][k](self.Z_x[:, :, j, k])
        
        # stack Z_t, Z_x into Z for later ref
        Z = torch.vstack((Z_t, self.Z_x))
        
        # a(x * z + b), shape: [B, W, N, M]
        A = self.activation(torch.einsum('bi,iojk->bojk', torch.hstack((t, x)), Z) + self.b)
        
        # NOT SURE BEYOND THIS
        
        V = torch.einsum()
        
        # shape [B, W, O, N, M]
        V = torch.zeros((batch_size, self.width, self.output_dim, self.N, self.M))
        
        # NOTE: may be able to vectorize in special cases
        for j in range(self.N):
            for d in range(self.output_dim):
                V[:, :, d, j, :] = A * self.p_array[j][d](Z)
        
        before_agg = torch.einsum('bwonm,wnm->bonm', V, self.W) # [B, O, N, M]
        outputs = before_agg.sum(dim=(-2, -1))  # [B, O], summed over last 2 dims
        
        return outputs / self.width
        
def main():
    
    full = False
    
    if full:
        N = 20 # number of lists of p's
        M = 2 # number of z expressions per p
        
        p_array = [[
            lambda Z: Z[0, :],
            lambda Z: Z[0, :],
            lambda Z: Z[0, :],
            lambda Z: Z[0, :],
            lambda Z: Z[0, :],
            lambda Z: Z[0, :]
        ] for _ in range(N)]
        
        z_array = [[
            lambda Z: torch.sqrt((Z ** 2).sum(0, keepdim=True)),
            lambda Z: -torch.sqrt((Z ** 2).sum(0, keepdim=True))
        ] for _ in range(N)]
        
        input_dim = 3
        
        m = Maxwell(input_dim=input_dim, p_array=p_array, z_array=z_array, width=100)
        
        data = grid_data(num_samples=10, spatial_dim=input_dim, mins=[0, 0, 0, 0], maxes=[1,1,1,1], restrict_time=True)[0]
        
        t = data[:, 0].unsqueeze(1)
        x = data[:, 1:]
        
        outs = m(t, x)
        
        print(outs.shape)
    
    else:
        input_dim = 3
        
        m = MaxwellSimple(input_dim=input_dim, width=100)
        
        data = grid_data(num_samples=10, spatial_dim=input_dim, mins=[0, 0, 0, 0], maxes=[1,1,1,1], restrict_time=True)[0]
        
        t = data[:, 0].unsqueeze(1)
        x = data[:, 1:]
        
        outs = m(t, x)
        
        print(outs.shape)
    

if __name__ == "__main__":
    main()