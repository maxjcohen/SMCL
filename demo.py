#!/usr/bin/env python
# coding: utf-8

import copy

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from smcl.smcl import SMCL
from oze.utils import plot_predictions
from oze.dataset import OzeDataset
from src.metrics import pi_metrics, compute_cost
from src.utils import (
    plot_range,
    uncertainty_estimation,
    LitProgressBar,
    boxplotprediction,
)

# Set manual seeds
torch.manual_seed(1)

# Matplotlib defaults
plt.rcParams.update({"font.size": 25, "figure.figsize": (25, 5)})

# PyTorch Lightning loading bar
bar = LitProgressBar()

# Dataset
PATH_DATASET = "datasets/data_oze.csv"
T = 24 * 7

# Model
D_EMB = 8
N = 200

# Training
BATCH_SIZE = 16
EPOCHS = 10
EPOCHS_SMCN = 100

## Dataset

df = pd.read_csv(PATH_DATASET)[5 * 24 :]
OzeDataset.preprocess(df)
df.sample(5)

_ = df[[*OzeDataset.input_columns, *OzeDataset.target_columns]].plot(
    subplots=True, figsize=(25, 40)
)
dataloader_train = DataLoader(
    OzeDataset(df, T=T, val=False), batch_size=BATCH_SIZE, num_workers=4, shuffle=True
)
dataloader_val = DataLoader(
    OzeDataset(df, T=T, val=True), batch_size=BATCH_SIZE, num_workers=4, shuffle=False
)

## Model

# We combine a generic input model (3 layered GRU) with a smc layer
class SMCM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_model = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=3
        )
        self.smcl = SMCL(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size,
            n_particles=100,
        )

    def forward(self, u, y=None):
        u_tilde = self.input_model(u)[0]
        y_hat = self.smcl(u_tilde, y)
        return y_hat

    def uncertainty_estimation(self, u, y=None, p=0.05, observation_noise=True):
        u_tilde = self.input_model(u)[0]
        return self.smcl.uncertainty_estimation(
            u_tilde, y=y, p=p, observation_noise=observation_noise
        )


D_IN = len(OzeDataset.input_columns)
D_OUT = len(OzeDataset.target_columns)
model = SMCM(input_size=D_IN, hidden_size=D_EMB, output_size=D_OUT)

## Traditional training
class LitClassicModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criteria = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        u, y = batch
        u = u.transpose(0, 1)
        y = y.transpose(0, 1)

        y_hat = self.model(u)
        loss = self.criteria(y, y_hat)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


train_model = LitClassicModule(model, lr=1e-2)
trainer = pl.Trainer(max_epochs=EPOCHS, gpus=1, callbacks=[bar])
trainer.fit(train_model, dataloader_train)

# Save pretrain parameters
params_pretrain = copy.deepcopy(model.state_dict())

# Compute cost (default to MSE) mean and variance
losses = compute_cost(model, dataloader_val)
print(f"MSE:\t{losses.mean():.2f} \pm {losses.var():.4f}")

plot_predictions(model, dataloader_train.dataset)
plot_predictions(model, dataloader_val.dataset)
plt.show()
