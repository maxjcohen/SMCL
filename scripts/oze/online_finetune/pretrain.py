import argparse
import datetime

import pytorch_lightning as pl

from src.litmodules import LitClassicModule
from src.utils import aim_fig_plot_ts
from ..classic import Experiment


class LitWeekEval(LitClassicModule):
    def validation_step(self, batch, batch_idx):
        u, y = batch
        y_hat = self.model(u)
        # Log global loss
        loss = self.criteria(y, y_hat)
        self.log("val_loss", loss)
        # Log weekly loss
        for week_idx, (week_observations, week_predictions) in enumerate(
            zip(y.transpose(0, 1), y_hat.transpose(0, 1))
        ):
            sample_idx = batch_idx * y.shape[1] + week_idx + 1
            loss = self.criteria(week_observations, week_predictions)
            self.log(f"val_loss_week{sample_idx}", loss)
            self.logger.experiment.track(
                aim_fig_plot_ts(
                    {
                        "observations": week_observations,
                        "predictions": week_predictions,
                    }
                ),
                name=f"week-{sample_idx}",
                epoch=self.current_epoch,
                context={"subset": "val"},
            )
        return y, y_hat


Experiment.exp_name = "oze_pretrain"
Experiment.dataset_kwargs = {
    "val_start": datetime.datetime(year=2021, month=5, day=1),
    "val_end": datetime.datetime(year=2021, month=5, day=1 + 7 * 4),
}
Experiment.LitModule = LitWeekEval
Experiment.monitor = "train_loss"


if __name__ == "__main__":
    args = argparse.Namespace(
        dataset_path="datasets/data_2020_2021.csv",
        T=24 * 7,
        d_emb=8,
        N=20,
        batch_size=16,
        num_workers=4,
        epochs=10,
        gpus=1,
        load_path=None,
    )

    exp = Experiment(args)

    exp.trainer.fit(exp.litmodule, datamodule=exp.datamodule)
    exp.trainer.validate(exp.litmodule, datamodule=exp.datamodule, ckpt_path="best")
