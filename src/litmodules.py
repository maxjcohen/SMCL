import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from smcl.smcl import SMCL

from .modules import GRUDropout
from .utils import flatten_batches, aim_fig_plot_ts


class LitSeqential(pl.LightningModule):
    def __init__(self, lr=1e-3, lookback_size: int = 24):
        super().__init__()
        self.save_hyperparameters()
        self.criteria = torch.nn.MSELoss()

    def forward(self, u, y):
        raise NotImplementedError

    def compute_loss(self, y, forecast):
        loss = self.criteria(
            y[self.hparams.lookback_size :], forecast[self.hparams.lookback_size :]
        )
        subset = "train" if self.training else "val"
        self.log(f"{subset}_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        u, y = batch
        forecast = self.forward(u, y)
        loss = self.compute_loss(y, forecast)
        if batch_idx == 0:
            self.logger.experiment.track(
                aim_fig_plot_ts(
                    {
                        "observations": y[:, 0],
                        "predictions": forecast[:, 0],
                    }
                ),
                name="batch-comparison",
                epoch=self.current_epoch,
                context={"subset": "train"},
            )
        return loss

    def validation_step(self, batch, batch_idx):
        u, y = batch
        # Remove y after lookback window
        observation_mask = torch.zeros_like(y, dtype=bool)
        observation_mask[self.hparams.lookback_size :] = True
        forecast = self.forward(u, y.masked_fill(observation_mask, float("nan")))
        self.compute_loss(y, forecast)
        if batch_idx == 0:
            self.logger.experiment.track(
                aim_fig_plot_ts(
                    {
                        "observations": y[:, 0],
                        "predictions": forecast[:, 0],
                    }
                ),
                name="batch-comparison",
                epoch=self.current_epoch,
                context={"subset": "val"},
            )
        return y[self.hparams.lookback_size :], forecast[self.hparams.lookback_size :]

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
        self._input_model_layers = nn.ModuleList(
            [
                nn.GRU(
                    input_size=input_size,
                    hidden_size=16,
                    num_layers=3,
                    dropout=0.2
                ),
                nn.GRU(
                    input_size=16,
                    hidden_size=hidden_size,
                    num_layers=1,
                ),
            ]
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

    def input_model(self, inputs):
        output = inputs
        for layer in self._input_model_layers:
            output, state = layer(output)
        return output, state

    @property
    def finetune(self):
        return self._finetune

    @finetune.setter
    def finetune(self, toogle: bool):
        # Freeze or thaw layer
        for param in self._input_model_layers.parameters():
            param.requires_grad = not toogle
        # Update finetune value
        self._finetune = toogle

    def forward(
        self,
        u: torch.Tensor,
        y: torch.Tensor | None = None,
        force_smc: bool = False,
        smc_average: bool = True,
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
        force_smc:
            If ``True``, will use the SMC layer regardless of the ``finetune``
            attribute. Default is ``False``.
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
        if force_smc or self.finetune:
            forecast = self.smcl(u_tilde, y)
            if smc_average:
                forecast = forecast.mean(-2)
        else:
            # Then through the deterministic emission layer
            initial_state = y[self.hparams.lookback_size - 1].unsqueeze(0).contiguous()
            initial_state /= 3  # Scale down between (almost) [-1, 1]
            u_tilde = u_tilde[self.hparams.lookback_size :]
            forecast = self.pretrain_toplayer(u_tilde, initial_state)[0] * 3
            # Prepend the lookback window
            forecast = torch.cat([y[: self.hparams.lookback_size], forecast], dim=0)
        return forecast

    def compute_loss(self, y, forecast):
        if not self.training or not self.finetune:
            return super().compute_loss(y, forecast)
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
        return loss

    def uncertainty_estimation(self, u, y=None, p=0.05, observation_noise=True):
        u_tilde = self.input_model(u)[0]
        return self.smcl.uncertainty_estimation(
            u_tilde, y=y, p=p, observation_noise=observation_noise
        )


class LitMCDropout(LitSeqential):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        lr: float | None = 1e-3,
    ):
        super().__init__(lr=lr)
        self.save_hyperparameters()
        self.input_model = GRUDropout(
            input_size=input_size, hidden_size=hidden_size, num_layers=3, dropout=0.001
        )
        self.emission = GRUDropout(
            input_size=hidden_size, hidden_size=output_size, num_layers=1, dropout=0.001
        )

    def forward(self, u, y):
        u_tilde = self.input_model(u)[0]
        # Then through the deterministic emission layer
        initial_state = y[self.hparams.lookback_size - 1].unsqueeze(0).contiguous()
        initial_state /= 3  # Scale down between (almost) [-1, 1]
        u_tilde = u_tilde[self.hparams.lookback_size :]
        forecast = self.emission(u_tilde, initial_state)[0] * 3
        # Prepend the lookback window
        forecast = torch.cat([y[: self.hparams.lookback_size], forecast], dim=0)
        return forecast


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
        self._input_model = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=3
        )
        self._emission = nn.GRU(
            input_size=hidden_size, hidden_size=output_size, num_layers=1
        )
        self.criteria = torch.nn.MSELoss()

    def forward(self, u_lookback, u, y_lookback, y=None):
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
        initial_state = y_lookback[-1].unsqueeze(0).contiguous()
        initial_state /= 3  # Scale down between (almost) [-1, 1]
        latents = self._input_model(u)[0]
        forecast = self._emission(latents, initial_state)[0] * 3
        return forecast
