import torch
from torch import nn

if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
    
class PCNN(nn.Module):
    
    def __init__(
        self,
        width: int = 10,
        c: float = 1.,
        input_dim: int = 1,
        output_dim: int = 1,
        activation: str = 'relu',
        # dropout_val: float = 0.1
        init: str = 'normal'
    ):
        super().__init__()
        
        self.width = width
        self.c = c
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.dropout_val = dropout_val
        self.init = init
        
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
        elif activation == 'sine':
            self.activation = lambda x: torch.sin(x)
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'cosine':
            self.activation = lambda x: torch.cos(x)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f'Unknown activation: {activation}')
        
        # self.dropout = nn.Dropout(self.dropout_val)