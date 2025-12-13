import torch
import torch.nn as nn

class Q_Former(nn.Module):
    def __init__(self, n_queries, n_dims) -> None:
        super().__init__(Q_Former)
        self.queries = nn.Parameter(
            torch.randn(n_queries, n_dims)
        )

    def forward(self, x):
        return x
        
