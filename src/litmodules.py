import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from smcl.smcl import SMCL

from .utils import flatten_batches, aim_fig_plot_ts
from .modules import LSTMDropout


class LitSeqential(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.criteria = torch.nn.MSELoss()

    def forward(self, u, y):
        return self.model(u)

    def training_step(self, batch, batch_idx):
        u, y = batch
        y_hat = self.forward(u, y)
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
        y_hat = self.forward(u, y)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


class LitSMCModule(LitSeqential):
    """Sequential Monte Carlo Module.

    By combining an input model with either a deterministic emission law or a SMC layer,
    this module can take avdantage of huge datasets while still being able to infer
    model and observation noise.

    Parameters
    ----------
    input_size:
        Dimension of the dataset's inputs.
    hidden_size:
        Size of the embedded space of the module.
    output_size:
        Dimension of the dataset's outputs.
    N:
        Number of particules to propagate.
    lr:
        Learning rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        N: int,
        lr: float = 1e-3,
    ):
        super().__init__(lr=lr)
        self.save_hyperparameters()
        self.input_model = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=3, dropout=0.2
        )
        self.pretrain_toplayer = nn.GRU(
            input_size=hidden_size, hidden_size=output_size, num_layers=1
        )
        self.smcl = SMCL(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size,
            n_particles=N,
        )
        self.criteria = torch.nn.MSELoss()
        self._SGD_idx = 1
        self._finetune = False

    @property
    def finetune(self):
        return self._finetune

    @finetune.setter
    def finetune(self, toogle: bool):
        # Freeze or thaw layer
        for param in self.input_model.parameters():
            param.requires_grad = not toogle
        # Update finetune value
        self._finetune = toogle

    def forward(
        self, u: torch.Tensor, y: torch.Tensor, smc_average: bool = False
    ) -> torch.Tensor:
        """Computes a forward pass through the module.

        The behavior of thie function depends on the `finetune` state of the module.
        First, in either case, we compute `u_tilde` by passing inputs to the input
        model. Then, if `finetune` is set to `False`, we feed this latent vector to the
        deterministic emmision layer. Otherwise, if `finetune` is set to `True`, we
        compute the sequential forwarding pass of the SMC layer, then optionally average
        on the particules to get a coherent output dimension.

        Parameters
        ----------
        u:
            Inputs as tensor with shape `(T, batch_size, d_in)`.
        y:
            Outputs as tensor with shape `(T, batch_size, d_out)`.
        smc_average:
            If `true`, will average over particules when using the SMC layer. Default is
            `False`.

        Returns
        -------
        Tensor with shape:
         - If using the SMC layer and `smc_average` is `False` (default), then
         `(T, batch_size, N. d_out)`.
         - Otherwise `(T, batch_size, d_out)`.
        """
        # Forward pass through the input model
        u_tilde = self.input_model(u)[0]
        if self.finetune:
            netout = self.smcl(u_tilde, y)
            if smc_average:
                netout = netout.mean(-2)
            return netout
        # Then through the deterministic emission layer
        initial_state = y[0].unsqueeze(0).contiguous()
        initial_state /= 3  # Scale down between (almost) [-1, 1]
        return self.pretrain_toplayer(u_tilde, initial_state)[0] * 3

    def uncertainty_estimation(self, u, y=None, p=0.05, observation_noise=True):
        u_tilde = self.input_model(u)[0]
        return self.smcl.uncertainty_estimation(
            u_tilde, y=y, p=p, observation_noise=observation_noise
        )

    def training_step(self, batch, batch_idx):
        u, y = batch
        y_hat = self.forward(u, y, smc_average=True)
        if self.finetune:
            # Compute loss
            loss = self.smcl.compute_cost(y=y)
            # Update Sigma_x
            gamma = 1 / np.sqrt(self._SGD_idx)
            self.smcl.sigma_x2 = (
                1 - gamma
            ) * self.smcl.sigma_x2 + gamma * self.smcl.compute_sigma_x()
            self.smcl.sigma_y2 = (
                1 - gamma
            ) * self.smcl.sigma_y2 + gamma * self.smcl.compute_sigma_y(y=y)
            self._SGD_idx += 1
            self.log("train_smcm_loss", loss)
            self.log("train_sigma_x", self.smcl.sigma_x2.diag().mean())
            self.log("train_sigma_y", self.smcl.sigma_y2.diag().mean())
        else:
            loss = self.criteria(y, y_hat)
            self.log("train_loss", loss, on_step=False, on_epoch=True)
        # Log visuals
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
        y_hat = self.forward(u, y, smc_average=True)
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


class LitLSTM(LitSeqential):
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
        super().__init__(lr=lr)
        self.save_hyperparameters()
        self._input_model = nn.LSTMDropout(
            input_size=input_size, hidden_size=hidden_size, num_layers=3
        )
        self._emission = nn.LSTMDropout(
            input_size=hidden_size, hidden_size=output_size, num_layers=1
        )
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

    def forward(self, u, y):
        return self.model(u, initial_state=y[0])
