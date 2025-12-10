import torch
from torch import nn

class Wave(nn.Module):

    def __init__(
        self,
        width: int = 10,
        c: float = 1.,
        input_dim: int = 1,
        output_dim: int = 1,
        activation: str = 'relu'
    ):
        super().__init__()

        self.width = width
        self.c = c
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        activation = activation.lower()
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish' or activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'square':
            self.activation = lambda x: x**2
        else:
            self.activation = nn.ReLU()

        self.position_weight_plus = nn.Parameter(torch.randn((self.input_dim, self.width)))
        self.position_weight_minus = nn.Parameter(torch.randn((self.input_dim, self.width)))
        
        self.bias_plus = nn.Parameter(torch.zeros((1, self.width)))
        self.bias_minus = nn.Parameter(torch.zeros((1, self.width)))

        self.output_weight = nn.Linear(width, self.output_dim)
        
        nn.init.normal_(self.position_weight_plus.data, 0, 1)
        nn.init.normal_(self.position_weight_minus.data, 0, 1)
        nn.init.normal_(self.output_weight.weight, 0, 1)

    def forward(self, t, x):

        pos_out_plus = x @ self.position_weight_plus
        pos_out_minus = x @ self.position_weight_minus

        time_out_plus = self.c * (
            t @ torch.sqrt((self.position_weight_plus ** 2).sum(0, keepdim=True))
        )
        time_out_minus = - self.c * (
            t @ torch.sqrt((self.position_weight_minus ** 2).sum(0, keepdim=True))
        )

        return self.output_weight(
            self.activation(time_out_plus + pos_out_plus + self.bias_plus)
            + self.activation(time_out_minus + pos_out_minus + self.bias_minus)
        ) / self.width
        
class WaveSimplified(nn.Module):

    def __init__(
        self,
        width: int = 10,
        c: float = 1.,
        input_dim: int = 1,
        output_dim: int = 1,
        activation: str = 'relu'
    ):
        super().__init__()

        self.width = width
        self.c = c
        self.input_dim = input_dim
        self.output_dim = output_dim

        activation = activation.lower()
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish' or activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'square':
            self.activation = lambda x: x**2
        else:
            self.activation = nn.ReLU()

        # self.position_weight_plus = nn.Parameter(torch.randn((self.input_dim, self.width)))
        self.position_weight_plus = nn.Linear(self.input_dim, width)

        self.output_weight = nn.Linear(width, self.output_dim)
        
        nn.init.normal_(self.position_weight_plus.weight, 0., 1.)
        nn.init.zeros_(self.position_weight_plus.bias)
        nn.init.normal_(self.output_weight.weight, 0., 1.)
        nn.init.zeros_(self.output_weight.bias)

    def forward(self, t, x):

        # pos_out_plus = x @ self.position_weight_plus
        pos_out_plus = self.position_weight_plus(x)

        return self.output_weight(
            self.activation(pos_out_plus)
        ) / self.width
        


class ConicalWave(nn.Module):

    def __init__(
        self,
        width: int = 10,
        c: float = 1.,
        input_dim: int = 1,
        output_dim: int = 1,
        activation: str = 'relu'
    ):
        super().__init__()

        self.width = width
        self.c = c
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        activation = activation.lower()
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish' or activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'square':
            self.activation = lambda x: x**2
        else:
            self.activation = nn.ReLU()

        self.position_weight_plus = nn.Parameter(torch.randn((self.input_dim, self.width)))
        self.position_weight_minus = nn.Parameter(torch.randn((self.input_dim, self.width)))
        
        self.bias_plus = nn.Parameter(torch.zeros((1, self.width)))
        self.bias_minus = nn.Parameter(torch.zeros((1, self.width)))

        self.output_weight = nn.Linear(width, self.output_dim)
        
        nn.init.normal_(self.position_weight_plus.data, 0, 1)
        nn.init.normal_(self.position_weight_minus.data, 0, 1)
        nn.init.normal_(self.output_weight.weight, 0, 1)

    def forward(self, t, x):

        pos_out_plus = x @ self.position_weight_plus
        pos_out_minus = x @ self.position_weight_minus

        time_out_plus = self.c * (
            t @ torch.sqrt((self.position_weight_plus ** 2).sum(0, keepdim=True))
        )
        time_out_minus = - self.c * (
            t @ torch.sqrt((self.position_weight_minus ** 2).sum(0, keepdim=True))
        )

        return self.output_weight(
            self.activation(time_out_plus + pos_out_plus + self.bias_plus)
            + self.activation(time_out_minus + pos_out_minus + self.bias_minus)
        ) / self.width



if __name__ == "__main__":

    batch_size = 32
    width = 10
    c = 1.
    input_dim = 3
    output_dim = 1
    activation = nn.ELU() # TODO: is ReLU okay?

    # predict at some random times/places
    t, x = (
        torch.randn((batch_size, 1)), # time is 1D
        torch.randn((batch_size, input_dim))
    )

    model = Wave(
        width = width,
        c = c,
        input_dim = input_dim,
        output_dim = output_dim,
        activation = activation
    )

    output = model(t, x)

    print(f'model(t in {t.shape}, x in {x.shape}) -> output in {output.shape}')