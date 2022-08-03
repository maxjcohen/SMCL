import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .modules import SMCM
from .utils import flatten_batches, aim_fig_plot_ts


class LitClassicModule(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, N, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SMCM(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size, N=N
        )
        self.lr = lr
        self.criteria = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        u, y = batch
        y_hat = self.model(u)
        loss = self.criteria(y, y_hat)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        if batch_idx == 0:
            self.logger.experiment.track(
                aim_fig_plot_ts(
                    {
                        "observations": y[:, 0],
                        "predictions": y_hat[:, 0],
                    }
                ),
                name="batch-comparison",
                epoch=self.current_epoch,
                context={"subset": "train"},
            )
        return loss

    def validation_step(self, batch, batch_idx):
        u, y = batch
        y_hat = self.model(u)
        loss = self.criteria(y, y_hat)
        self.log("val_loss", loss)
        if batch_idx == 0:
            self.logger.experiment.track(
                aim_fig_plot_ts(
                    {
                        "observations": y[:, 0],
                        "predictions": y_hat[:, 0],
                    }
                ),
                name="batch-comparison",
                epoch=self.current_epoch,
                context={"subset": "val"},
            )
        return y, y_hat

    def validation_epoch_end(self, outputs):
        observations, predictions = map(flatten_batches, zip(*outputs))
        self.logger.experiment.track(
            aim_fig_plot_ts(
                {
                    "observations": observations,
                    "predictions": predictions,
                }
            ),
            name="full-comparison",
            epoch=self.current_epoch,
            context={"subset": "val"},
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class LitLSTM(LitClassicModule):
    """Traditional LSTM model.

    This module combines an LSTM input model, as well as a GRU emission function. The
    inital state of this last layer must be set explicitly for our use case. For this
    reason, it has been defined as a GRU, in order to avoid dealing with multiple latent
    states.

    Parameters
    ----------
    input_size:
        Dimension of the input sequence.
    hidden_size:
        Chosen dimension of the latent space.
    output_size:
        Dimension of the output sequence.
    lr:
        Learning rate. Default is `1e-3`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        lr: float | None = 1e-3,
    ):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self._input_model = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=3
        )
        self._emission = nn.GRU(
            input_size=hidden_size, hidden_size=output_size, num_layers=1
        )
        self.lr = lr
        self.criteria = torch.nn.MSELoss()

    def model(self, x: torch.Tensor, initial_state: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass for the module.

        The emission GRU layer's hidden state is intialized with `initial_state`.
        Because the output of this layer is constrained to `[-1, 1]`, we apply a simple
        multiplicative factor.

        Parameters
        ----------
        x:
            Input tensor with shape `(T, BS, d_in)`.
        initial_state:
            Initial hidden state of the emission GRU layer with shape `(BS, d)`.

        Returns
        -------
        Predicted vector with shape `(T, BS, d)`, hopefully carrying a gradient.
        """
        initial_state = initial_state.unsqueeze(0).contiguous()
        initial_state /= 3  # Scale down between (almost) [-1, 1]
        return self._emission(self._input_model(x)[0], initial_state)[0] * 3

    def training_step(self, batch, batch_idx):
        u, y = batch
        y_hat = self.model(u, y[0])
        loss = self.criteria(y, y_hat)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        if batch_idx == 0:
            self.logger.experiment.track(
                aim_fig_plot_ts(
                    {
                        "observations": y[:, 0],
                        "predictions": y_hat[:, 0],
                    }
                ),
                name="batch-comparison",
                epoch=self.current_epoch,
                context={"subset": "train"},
            )
        return loss

    def validation_step(self, batch, batch_idx):
        u, y = batch
        y_hat = self.model(u, y[0])
        loss = self.criteria(y, y_hat)
        self.log("val_loss", loss)
        if batch_idx == 0:
            self.logger.experiment.track(
                aim_fig_plot_ts(
                    {
                        "observations": y[:, 0],
                        "predictions": y_hat[:, 0],
                    }
                ),
                name="batch-comparison",
                epoch=self.current_epoch,
                context={"subset": "val"},
            )
        return y, y_hat


class LitSMCModule(LitClassicModule):
    _SGD_idx = 1

    def training_step(self, batch, batch_idx):
        u, y = batch
        # Forward pass
        self.model(u=u, y=y)
        # Compute loss
        loss = self.model.smcl.compute_cost(y=y)
        # Update Sigma_x
        gamma = 1 / np.sqrt(self._SGD_idx)
        self.model.smcl.sigma_x2 = (
            1 - gamma
        ) * self.model.smcl.sigma_x2 + gamma * self.model.smcl.compute_sigma_x()
        self.model.smcl.sigma_y2 = (
            1 - gamma
        ) * self.model.smcl.sigma_y2 + gamma * self.model.smcl.compute_sigma_y(y=y)
        self._SGD_idx += 1
        self.log("train_smcm_loss", loss, on_step=False, on_epoch=True)
        self.log("train_sigma_x", self.model.smcl.sigma_x2.diag().mean())
        self.log("train_sigma_y", self.model.smcl.sigma_y2.diag().mean())
        return loss

    def configure_optimizers(self):
        # We only optimize SMCL parameters
        optimizer = torch.optim.Adam(self.model.smcl.parameters(), lr=self.lr)
        return optimizer
