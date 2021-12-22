#!/usr/bin/env python
# coding: utf-8

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger

from smcl.smcl import SMCL
from oze.utils import plot_predictions
from oze.dataset import OzeDataset
from src.litmodules import LitSMCModule, LitClassicModule
from src.metrics import compute_cost, cumulative_cost
from src.utils import (
    plot_range,
    uncertainty_estimation,
    boxplotprediction,
)

# Params
class args:
    dataset_path = "datasets/data_2020_2021.csv"
    T = 24 * 7

    # Model
    d_emb = 8
    N = 20

    # Training
    batch_size = 16
    epochs = 10
    epochs_smcn = 10

    train = ["smcl"]

    save_path = None
    load_path = "weights/pretrain.pt"


# Set manual seeds
torch.manual_seed(1)

# Matplotlib defaults
plt.rcParams.update({"font.size": 25, "figure.figsize": (25, 5)})


def train(train_model, exp_name, args):
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=1,
        logger=AimLogger(experiment=exp_name, system_tracking_interval=None),
    )
    trainer.fit(train_model, dataloader_train, val_dataloaders=dataloader_val)


# Load dataset
df = pd.read_csv(args.dataset_path)[5 * 24 :]
OzeDataset.preprocess(df)
# Plot input and outputs
# TODO move to logger
_ = df[[*OzeDataset.input_columns, *OzeDataset.target_columns]].plot(
    subplots=True,
    figsize=(25, 3 * (len(OzeDataset.input_columns + OzeDataset.target_columns))),
)


def collate_fn(batch):
    u, y = list(zip(*batch))
    u = torch.stack(u).transpose(0, 1)
    y = torch.stack(y).transpose(0, 1)
    return u, y


# Define dataloaders
dataloader_train = DataLoader(
    OzeDataset(df, T=args.T, val=False),
    batch_size=args.batch_size,
    num_workers=4,
    shuffle=True,
    collate_fn=collate_fn,
)
dataloader_val = DataLoader(
    OzeDataset(df, T=args.T, val=True),
    batch_size=args.batch_size,
    num_workers=4,
    shuffle=False,
    collate_fn=collate_fn,
)


# We combine a generic input model (3 layered GRU) with a smc layer
class SMCM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_model = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=3, dropout=0.2
        )
        self.smcl = SMCL(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size,
            n_particles=args.N,
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


d_in = len(OzeDataset.input_columns)
d_out = len(OzeDataset.target_columns)

model = SMCM(input_size=d_in, hidden_size=args.d_emb, output_size=d_out)
if args.load_path is not None:
    model.load_state_dict(torch.load(args.load_path))
    print("Loaded model weights.")

if "classic" in args.train:
    train_model = LitClassicModule(model, lr=1e-3)
    exp_name = "pretrain"
    train(train_model, exp_name, args)
elif "smcl" in args.train:
    train_model = LitSMCModule(model, lr=1e-3)
    exp_name = "smcl"
    train(train_model, exp_name, args)

if args.save_path is not None:
    torch.save(model.state_dict(), args.save_path)

plot_predictions(model, dataloader_train.dataset)
plot_predictions(model, dataloader_val.dataset)

plt.show()
