"""Export oze dataset to csv format.

The Oze dataset is retrieved in npy format from Maurice's pipe. This script simply
converts his specific data structure in a classic csv.
"""
#!/usr/bin/env python
# coding: utf-8

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

export_path = "datasets/data_2020_2021.csv"

dataset_paths = [
    "datasets/Le Stephenson_data_from_1_1_2020_for_365_days.npy",
    "datasets/Le Stephenson_data_from_1_1_2021_for_279_days.npy",
]

columns = [
    "IBEAM_H",
    "IBEAM_N",
    "IDIFF_H",
    "GK",
    "RHUM",
    "IGLOB_H",
    "temperature_exterieure",
    "temperature_interieure",
    "taux_co2",
    "humidite",
    "cta_temperature",
    "electricite",
]


def load_npy(path: Path) -> pd.DataFrame:
    ds = np.load(path, allow_pickle=True).item(0)
    return pd.DataFrame({**ds["OZE_data"], **ds["meteo_data"]})


# Concat loaded npy in a single dataframe
df = pd.concat(objs=[load_npy(path) for path in dataset_paths], ignore_index=True)[
    columns
]


# Add `date` column
start_date = datetime.datetime(year=2020, month=1, day=1)
df["date"] = df.index.map(
    lambda t: (start_date + datetime.timedelta(hours=t)).timestamp()
)

df.to_csv(export_path, index=False)

# Plot results
_ = pd.read_csv(export_path).plot(subplots=True, figsize=(25, 40))
