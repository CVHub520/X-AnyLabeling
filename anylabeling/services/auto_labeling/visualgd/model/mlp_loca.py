from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        activation: nn.Module,
    ):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
