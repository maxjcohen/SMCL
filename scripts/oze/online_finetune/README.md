# Online finetuning experiment

This experiment aims at evaluating the gain from finetuning the classic model (i.e. a simple LSTM) for each new week of data. This example highlights that even for such a simple model, results are not straight forward. Because of this, we did not pursue to evaluate this methodology with SMCL.

## Run the experiment

1. Pretrain the model by running:
```bash
python -m scripts.oze.online_finetune.pretrain
```

2. Edit finetune script to match the checkpoint path and `WEEK_IDX` number
```python
WEEK_IDX = 1
# ...
load_path="path/to/checkpoint.ckpt",
```

3. Run the finetuning script
```bash
python -m scripts.oze.online_finetune.finetune
```

Repeat steps 2-3 as many times as necessary with different week indexes. With the current time ranges, finetuning for week 1 and 2 should be enough to see that no easy conclusion can be drawn this way.
