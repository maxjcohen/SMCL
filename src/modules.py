import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli


class GRUDropout(nn.Module):
    """GRU Dropout module.

    This module defines a stack of GRU Dropout layers, see the module
    `_RNNDropoutLayer` for mode information. It is a drop-in replacement for torch's
    `GRU` module.

    Parameters
    ----------
    dropout:
        Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        rnn_layers = [
            _RNNDropoutLayer(
                input_size=input_size, hidden_size=hidden_size, dropout=dropout
            )
        ]
        for _ in range(num_layers - 1):
            rnn_layers.append(
                _RNNDropoutLayer(
                    input_size=hidden_size, hidden_size=hidden_size, dropout=dropout
                )
            )
        self.rnn_layers = nn.ModuleList(rnn_layers)

    def forward(self, inputs, h_0=None):
        h_0 = h_0 if h_0 is not None else [None for _ in range(len(self.rnn_layers))]
        outputs = inputs
        for (layer, h_0_layer) in zip(self.rnn_layers, h_0):
            outputs, h_k = layer(outputs, h_0=h_0_layer)
        return outputs, h_k


class _RNNDropoutLayer(nn.Module):
    """RNN Dropout layer.

    RNN layer with MC Dropout regularization, as introduced in `Gal, Y., & Ghahramani,
    Z. (2016). A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks. NIPS` ([arxiv](https://arxiv.org/pdf/1512.05287.pdf)).

    In order to preserve compatibility with torch's API, while implementing a clear and
    easy to review function, we define the logic of the MC Dropout in this module and
    use the `GRUDropout` module publicly.
    """

    def __init__(
        self, input_size: int, hidden_size: int, dropout: float = 0.2, **kwargs
    ):
        super().__init__(**kwargs)
        self.rnn_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.p_dropout = dropout

    def dropout(self, tensor, mask, p):
        return tensor.masked_fill(mask, 0) / (1 - p)

    def forward(self, x, h_0=None):
        """
        X: (Seq_len, batch_size, dim)
        """
        batch_size = x.shape[1]
        # Initialize hidden states
        h_k = (
            h_0
            if h_0 is not None
            else torch.zeros(batch_size, self.hidden_size, device=x.device)
        )
        # Generate dropout masks
        dropout_mask_input = (
            Bernoulli(torch.full(x.shape[1:], self.p_dropout))
            .sample()
            .to(dtype=bool, device=x.device)
        )
        dropout_mask_hidden = (
            Bernoulli(torch.full(h_k.shape, self.p_dropout))
            .sample()
            .to(dtype=bool, device=h_k.device)
        )
        # Loop through time
        outputs = []
        for x_k in x:
            if self.training:
                h_k = self.dropout(h_k, dropout_mask_hidden, p=self.p_dropout)
                x_k = self.dropout(x_k, dropout_mask_input, p=self.p_dropout)
            # Compute RNN cell
            h_k = self.rnn_cell(x_k, h_k)
            outputs.append(h_k)
        return torch.stack(outputs, dim=0), h_k

    @property
    def hidden_size(self):
        return self.rnn_cell.hidden_size
