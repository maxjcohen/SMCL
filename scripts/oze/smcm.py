import argparse
import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from aim.pytorch_lightning import AimLogger
from ozedata import OzeDataModule

from src.litmodules import LitSMCModule
from .classic import Experiment as PretrainExperiment


class Experiment(PretrainExperiment):
    exp_name = "oze_smcm"
    LitModule = LitSMCModule
    monitor = "val_loss"


if __name__ == "__main__":
    args = argparse.Namespace(
        dataset_path="datasets/data_2020_2021.csv",
        T=24 * 7,
        d_emb=8,
        N=20,
        batch_size=16,
        num_workers=4,
        epochs=30,
        gpus=1,
        load_path="checkpoints/oze_pretrain/last.ckpt",
    )

    exp = Experiment(args)

    exp.trainer.fit(exp.litmodule, datamodule=exp.datamodule)
