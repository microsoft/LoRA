from torch import nn
from transformers.activations import get_activation


class Adapter(nn.Module):
    def __init__(self, dim, r, act):
        super().__init__()
        self.adapter_A = nn.Linear(dim, r)
        self.act = get_activation(act)
        self.adapter_B = nn.Linear(r, dim)

    def forward(self, x, residual):
        result = self.adapter_A(x)
        result = self.act(result)
        result = self.adapter_B(result)
        return result + residual
