import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli


class LSTMDropoutLayer(nn.Module):

    """LSTM Dropout layer

    From the paper "A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks."
    """

    def __init__(self, input_size: int, hidden_size: int, p: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.rnn_cell = nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size, **kwargs
        )
        self.p_dropout = p

    def dropout(self, tensor, mask, p):
        return tensor.masked_fill(mask, 0) / (1 - p)

    def forward(self, x):
        """
        X: (Seq_len, batch_size, dim)
        """
        batch_size = x.shape[1]
        # Initialize hidden states
        hx = torch.randn(batch_size, self.hidden_size, device=x.device)
        cx = torch.randn(batch_size, self.hidden_size, device=x.device)
        # Generate dropout masks
        dropout_mask_hx = (
            Bernoulli(torch.full(hx.shape, self.p_dropout))
            .sample()
            .to(dtype=bool, device=hx.device)
        )
        dropout_mask_cx = (
            Bernoulli(torch.full(cx.shape, self.p_dropout))
            .sample()
            .to(dtype=bool, device=cx.device)
        )
        # Loop through time
        outputs = []
        for x_k in x:
            # Add dropout for LSTM Dropout
            hx = self.dropout(hx, dropout_mask_hx, p=self.p_dropout)
            cx = self.dropout(cx, dropout_mask_cx, p=self.p_dropout)
            # Compute RNN cell
            hx, cx = self.rnn_cell(x_k, (hx, cx))
            outputs.append(hx)
        return torch.stack(outputs, dim=0)

    @property
    def hidden_size(self):
        return self.rnn_cell.hidden_size
