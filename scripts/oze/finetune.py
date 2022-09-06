import argparse
from .pretrain import Experiment as ExperimentPretrain


class Experiment(ExperimentPretrain):
    exp_name = "oze_finetune"


if __name__ == "__main__":
    args = argparse.Namespace(
        dataset_path="datasets/data_2020_2021.csv",
        T=48,
        d_emb=3,
        N=100,
        batch_size=32,
        num_workers=4,
        epochs=100,
        gpus=1,
        lr=3e-3,
        load_path="checkpoints/oze_pretrain/2022_09_05__170751.ckpt",
        finetune=True,
        logger=True,
    )

    exp = Experiment(args)

    exp.trainer.fit(exp.litmodule, datamodule=exp.datamodule)
    exp.trainer.validate(exp.litmodule, datamodule=exp.datamodule, ckpt_path="best")
