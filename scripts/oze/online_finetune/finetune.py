import argparse
import datetime

from ozedata import OzeDataModule
from ozedata.oze.dataset import OzeDataset

from .pretrain import Experiment, LitWeekEval

WEEK_IDX = 1

Experiment.exp_name = f"oze_finetune_week{WEEK_IDX}"
# Select a single week to train on
start_day = 1 + 7 * (WEEK_IDX - 1)
Experiment.dataset_kwargs = {
    "train_start": datetime.datetime(year=2021, month=5, day=start_day),
    "val_start": datetime.datetime(year=2021, month=5, day=start_day + 7),
}
# Disable rolling dataset as there is only one week to train on
Experiment.DataModule.DatasetRolling = OzeDataset


if __name__ == "__main__":
    args = argparse.Namespace(
        dataset_path="datasets/data_2020_2021.csv",
        T=24 * 7,
        d_emb=8,
        N=20,
        batch_size=16,
        num_workers=4,
        epochs=15,
        gpus=1,
        load_path="checkpoints/oze_pretrain/2022_08_01__165408.ckpt",
    )

    # Train
    exp = Experiment(args)
    exp.datamodule.setup()
    exp.trainer.fit(exp.litmodule, train_dataloaders=exp.datamodule.train_dataloader())

    # Validate on all validation weeks
    exp.datamodule = OzeDataModule(
        dataset_path=args.dataset_path,
        T=args.T,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_start=datetime.datetime(year=2021, month=5, day=1),
        val_end=datetime.datetime(year=2021, month=5, day=1 + 7 * 4),
    )
    exp.trainer.validate(exp.litmodule, datamodule=exp.datamodule)
