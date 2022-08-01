import numpy as np
import torch
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
