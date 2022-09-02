import argparse
import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from aim.pytorch_lightning import AimLogger
from ozedata import ETDataModule

from src.litmodules import LitSMCModule


class Experiment:
    exp_name = "ett_pretrain"
    LitModule = LitSMCModule
    d_in = 6
    d_out = 1
    monitor = "val_loss"

    def __init__(self, args):
        self.datamodule = ETDataModule(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            forecast_size=args.T,
        )

        if args.load_path:
            self.litmodule = self.LitModule.load_from_checkpoint(
                args.load_path, lr=args.lr
            )
        else:
            self.litmodule = self.LitModule(
                input_size=self.d_in,
                hidden_size=args.d_emb,
                output_size=self.d_out,
                N=args.N,
                lr=args.lr,
            )
        self.litmodule.finetune = args.finetune

        if not args.logger:
            self.logger = False
        else:
            self.logger = AimLogger(
                experiment=self.exp_name, system_tracking_interval=None
            )
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
        dataset_path="datasets/ETTh1.csv",
        T=48,
        d_emb=16,
        N=100,
        batch_size=32,
        num_workers=4,
        epochs=30,
        gpus=1,
        lr=3e-4,
        load_path=None,
        finetune=False,
        logger=True,
    )

    exp = Experiment(args)

    exp.trainer.fit(exp.litmodule, datamodule=exp.datamodule)
