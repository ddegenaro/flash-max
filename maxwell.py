from typing import Callable

import torch
from torch import nn

from utils import PCNN

class Maxwell(PCNN):

    def __init__(
        self,
        width: int = 10,
        c: float = 1.,
        input_dim: int = 1,
        output_dim: int = 1,
        activation: str = 'relu',
        dropout_val: float = 0.1,
        p_array: list[Callable] = []
    ):
        super().__init__(
            width=width,
            c=c,
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            dropout_val=dropout_val
        )
        
        self.num_ps = len(p_array)
        
        self.position_weight_plus = nn.ModuleList(
            [nn.Parameter(torch.randn((self.input_dim, self.width))) for _ in range(self.num_ps)]
        )
        self.position_weight_minus = nn.ModuleList(
            [nn.Parameter(torch.randn((self.input_dim, self.width))) for _ in range(self.num_ps)]
        )
        
        self.bias_plus = nn.ModuleList(
            [nn.Parameter(torch.zeros((1, self.width))) for _ in range(self.num_ps)]
        )
        self.bias_minus = nn.ModuleList(
            [nn.Parameter(torch.zeros((1, self.width))) for _ in range(self.num_ps)]
        )

        self.output_weight_plus = nn.ModuleList(
            [nn.Linear(width, self.output_dim) for _ in range(self.num_ps)]
        )
        self.output_weight_minus = nn.ModuleList(
            [nn.Linear(width, self.output_dim) for _ in range(self.num_ps)]
        )
        
        for i in range(self.num_ps):
            nn.init.normal_(self.position_weight_plus[i].data, 0, 1)
            nn.init.normal_(self.position_weight_minus[i].data, 0, 1)
            nn.init.normal_(self.output_weight_plus[i].weight, 0, 1)
            nn.init.normal_(self.output_weight_minus[i].weight, 0, 1)
    
    def forward(self, t, x):
        
        pos_out_plus = x @ self.position_weight_plus
        pos_out_minus = x @ self.position_weight_minus
        
        if self.dropout_val > 0:
            pos_out_plus = self.dropout(pos_out_plus)
            pos_out_minus = self.dropout(pos_out_minus)

        time_out_plus = self.c * (
            t @ torch.sqrt((self.position_weight_plus ** 2).sum(0, keepdim=True))
        )
        time_out_minus = - self.c * (
            t @ torch.sqrt((self.position_weight_minus ** 2).sum(0, keepdim=True))
        )

        return (
            self.output_weight_plus(
                self.activation(time_out_plus + pos_out_plus + self.bias_plus)
            ) + self.output_weight_minus(
                self.activation(time_out_minus + pos_out_minus + self.bias_minus)
            )
        ) / self.width