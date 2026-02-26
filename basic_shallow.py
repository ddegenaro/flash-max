import torch
from torch import nn

from utils import PCNN

class Simple(PCNN):

    def __init__(
        self,
        width: int = 10,
        c: float = 1.,
        input_dim: int = 1,
        output_dim: int = -1,
        activation: str = 'relu',
        # dropout_val: float = 0.1,
        init = 'kaiming'
    ):
        
        super().__init__(
            width=width,
            c=c,
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            # dropout_val=dropout_val
            init=init
        )
        
        self.Z = nn.Parameter(torch.zeros((input_dim, self.width)))
        self.b = nn.Parameter(torch.zeros((1, self.width)))
        self.W = nn.Parameter(torch.zeros((self.width, self.output_dim)))
        
        if self.init == 'kaiming':
            nn.init.kaiming_normal_(self.Z.data)
            nn.init.kaiming_normal_(self.W.data.T)
        elif self.init == 'normal':
            nn.init.normal_(self.Z.data)
            nn.init.normal_(self.W.data)
        
    def forward(self, t=None, x=None):
        
        return (
            self.activation(x @ self.Z + self.b) * self.W
        ) @ (
            -self.Z[0] * self.Z[2]
        )