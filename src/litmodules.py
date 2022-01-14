import numpy as np
import torch
import pytorch_lightning as pl


class LitClassicModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criteria = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        u, y = batch
        y_hat = self.model(u)
        loss = self.criteria(y, y_hat)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        u, y = batch
        y_hat = self.model(u)
        loss = self.criteria(y, y_hat)
        self.log("validation_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class LitSMCModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.MSE = torch.nn.MSELoss()
        self._SGD_idx = 1

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
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("sigma_x", self.model.smcl.sigma_x2.diag().mean(), on_step=False, on_epoch=True)
        self.log("sigma_y", self.model.smcl.sigma_y2.diag().mean(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        u, y = batch
        # Forward pass
        self.model(u=u, y=y)
        # Compute loss
        loss = self.model.smcl.compute_cost(y=y)
        self.log("validation_loss", loss)

    def configure_optimizers(self):
        # We only optimize SMCL parameters
        optimizer = torch.optim.Adam(self.model.smcl.parameters(), lr=self.lr)
        return optimizer
