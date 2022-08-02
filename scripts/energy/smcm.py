import argparse
import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from aim.pytorch_lightning import AimLogger
from ozedata import EnergyDataModule

from src.litmodules import LitSMCModule
from .pretrain import Experiment as PretrainExperiment


class Experiment(PretrainExperiment):
    exp_name = "energy_smcm"
    LitModule = LitSMCModule
    monitor = "train_smcm_loss"


if __name__ == "__main__":
    args = argparse.Namespace(
        dataset_path="datasets/energydata_complete.csv",
        T=24 * 6,
        d_emb=8,
        N=200,
        batch_size=8,
        num_workers=4,
        epochs=100,
        gpus=1,
        load_path="checkpoints/energy_pretrain/2022_08_02__124514.ckpt",
    )

    exp = Experiment(args)

    exp.trainer.fit(exp.litmodule, datamodule=exp.datamodule)
