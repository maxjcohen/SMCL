import torch
import torch.nn as nn

from smcl.smcl import SMCL
from src.lstm_dropout import LSTMDropoutLayer


class LSTMDropout(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

        self.input_model = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=3, dropout=0.2
        )
        self.lstm_dropout = LSTMDropoutLayer(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, u: torch.Tensor, y=None) -> torch.Tensor:
        u_tilde = self.input_model(u)[0]
        y_hat = self.lstm_dropout(u_tilde)
        y_hat = self.linear(y_hat)
        return y_hat

    def uncertainty_estimation(self, u, y=None, N=10, p=0.05):
        u_tilde = self.input_model(u)[0]
        preds = torch.stack([self(u) for _ in range(N)], dim=2)
        return preds
