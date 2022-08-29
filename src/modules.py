import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli


class LSTMDropout(nn.Module):
    """LSTM Dropout module.

    This module defines a stack of LSTM Dropout layers, see the module
    `_LSTMDropoutLayer` for mode information. It is a drop-in replacement for torch's
    `LSTM` module.

    Parameters
    ----------
    p:
        Dropout probability
    """

    def __init__(self, input_size, hidden_size, num_layers: int, p: float, **kwargs):
        super().__init__()
        lstm_layers = [
            _LSTMDropoutLayer(input_size=input_size, hidden_size=hidden_size, p=p)
        ]
        for _ in range(num_layers - 1):
            lstm_layers.append(
                _LSTMDropoutLayer(input_size=hidden_size, hidden_size=hidden_size)
            )
        self.lstm_layers = nn.ModuleList(lstm_layers)

    def forward(self, inputs):
        outputs = inputs
        for layer in self.lstm_layers:
            outputs, (hx, cx) = layer(outputs)
        return outputs, (hx, cx)


class _LSTMDropoutLayer(nn.Module):
    """LSTM Dropout layer.

    LSTM layer with MC Dropout regularization, as introduced in `Gal, Y., & Ghahramani,
    Z. (2016). A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks. NIPS` ([arxiv](https://arxiv.org/pdf/1512.05287.pdf)).

    In order to preserve compatibility with torch's API, while implementing a clear and
    easy to review function, we define the logic of the MC Dropout in this module and
    use the `LSTMDropout` module publicly.
    """

    def __init__(self, input_size: int, hidden_size: int, p: float = 0.2, **kwargs):
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
        return torch.stack(outputs, dim=0), (hx, cx)

    @property
    def hidden_size(self):
        return self.rnn_cell.hidden_size
