import argparse
from .pretrain import Experiment as ExperimentPretrain


class Experiment(ExperimentPretrain):
    exp_name = "ett_finetune"


if __name__ == "__main__":
    args = argparse.Namespace(
        dataset_path="datasets/ETTh1.csv",
        T=48,
        d_emb=3,
        N=100,
        batch_size=32,
        num_workers=4,
        epochs=10,
        gpus=1,
        lr=1e-3,
        load_path="checkpoints/saves/ett_pretrain.ckpt",
        finetune=True,
        logger=True,
    )

    exp = Experiment(args)

    exp.trainer.fit(exp.litmodule, datamodule=exp.datamodule)
    exp.trainer.validate(exp.litmodule, datamodule=exp.datamodule, ckpt_path="best")
