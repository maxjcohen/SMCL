import argparse
import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from aim.pytorch_lightning import AimLogger
from ozedata import OzeDataModule

from src.litmodules import LitClassicModule


class Experiment:
    exp_name = "oze_classic"
    d_in = 2
    d_out = 1
    LitModule = LitClassicModule
    DataModule = OzeDataModule
    dataset_kwargs = {}
    monitor = "val_loss"

    def __init__(self, args):
        self.datamodule = self.DataModule(
            dataset_path=args.dataset_path,
            T=args.T,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            **self.dataset_kwargs,
        )

        if args.load_path:
            self.litmodule = self.LitModule.load_from_checkpoint(args.load_path)
        else:
            self.litmodule = self.LitModule(
                input_size=self.d_in,
                hidden_size=args.d_emb,
                output_size=self.d_out,
                N=args.N,
            )

        self.logger = AimLogger(experiment=self.exp_name, system_tracking_interval=None)
        self.logger.experiment["hparams"] = vars(args)
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path("checkpoints") / self.exp_name,
            filename=f"{datetime.datetime.now().strftime('%Y_%m_%d__%H%M%S')}",
            monitor=self.monitor,
            save_last=True,
        )
        self.trainer = pl.Trainer(
            max_epochs=args.epochs,
            gpus=args.gpus,
            logger=self.logger,
            callbacks=[checkpoint_callback],
            log_every_n_steps=1,
        )


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
        load_path=None,
    )

    exp = Experiment(args)

    exp.trainer.fit(exp.litmodule, datamodule=exp.datamodule)
