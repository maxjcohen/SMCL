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
from src.modules import SMCM, LSTMDropout
from src.litmodules import LitSMCModule, LitClassicModule
from src.metrics import compute_cost, cumulative_cost
from src.utils import (
    plot_range,
    uncertainty_estimation,
    boxplotprediction,
)

# Params
class args:
    dataset_path = "datasets/energydata_complete.csv"
    T = 6*24

    # Model
    d_emb = 8
    N = 100

    # Training
    batch_size = 8
    epochs = 100
    epochs_smcn = 10
    lr=2e-3

    train = ["classic"]

    save_path = "weights/energy_pretrain.pt"
    load_path = None #"weights/pretrain.pt"


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
df = pd.read_csv(args.dataset_path)
class EnergyDataset(OzeDataset):
    input_columns = [
        "Appliances",
        "lights",
        "T_out",
        # "Press_mm_hg",
        # "RH_out",
        # "Windspeed",
        # "Visibility",
        # "Tdewpoint",
    ]
    target_columns = [
        "T1",
        # "RH_1",
    ]

    @staticmethod
    def preprocess(df):
        df["val"] = np.arange(len(df)) > 15000


EnergyDataset.preprocess(df)


def collate_fn(batch):
    u, y = list(zip(*batch))
    u = torch.stack(u).transpose(0, 1)
    y = torch.stack(y).transpose(0, 1)
    return u, y


# Define dataloaders
dataloader_train = DataLoader(
    EnergyDataset(df, T=args.T, val=False),
    batch_size=args.batch_size,
    num_workers=4,
    shuffle=True,
    collate_fn=collate_fn,
)
dataloader_val = DataLoader(
    EnergyDataset(df, T=args.T, val=True),
    batch_size=args.batch_size,
    num_workers=4,
    shuffle=False,
    collate_fn=collate_fn,
)


# Load model
d_in = len(EnergyDataset.input_columns)
d_out = len(EnergyDataset.target_columns)
model = SMCM(input_size=d_in, hidden_size=args.d_emb, output_size=d_out, N=args.N)

if __name__ == "__main__":
    if args.load_path is not None:
        model.load_state_dict(torch.load(args.load_path))
        print("Loaded model weights.")

    if "classic" in args.train:
        train_model = LitClassicModule(model, lr=args.lr)
        exp_name = "energy-pretrain"
        train(train_model, exp_name, args)
    elif "smcl" in args.train:
        train_model = LitSMCModule(model, lr=args.lr)
        exp_name = "energy-smcl"
        train(train_model, exp_name, args)

    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)

    plot_predictions(model, dataloader_train.dataset)
    plot_predictions(model, dataloader_val.dataset)

    plt.show()
