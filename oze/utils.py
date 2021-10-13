import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_predictions(model: callable, dataloader, target_columns: list):
    with torch.no_grad():
        netout = model(dataloader.dataset._u.unsqueeze(0).transpose(0, 1))

    netout = netout.numpy()[:, 0]

    observations = dataloader.dataset.y.numpy()
    netout = dataloader.dataset.rescale(netout, "observation")

    n_plots = netout.shape[1]
    plt.figure(figsize=(25, n_plots * 5))
    for idx_plot, (observation, prediction) in enumerate(zip(observations.T, netout.T)):

        plt.subplot(n_plots, 1, idx_plot + 1)
        plt.plot(observation, label="observation", lw=3)
        plt.plot(prediction, label="prediction")
        plt.title(target_columns[idx_plot])
        plt.legend()


def plot_daily_predictions(model, dataloader):
    n_days = len(dataloader.dataset)
    targets = []
    outputs = []
    for idx_sample in range(n_days):
        sample_inputs, sample_targets = dataloader.dataset[idx_sample]
        targets.append(sample_targets)
        with torch.no_grad():
            outputs.append(model(sample_inputs.unsqueeze(1))[:, 0])

    targets = dataloader.dataset.rescale(torch.cat(targets), "observation")
    outputs = dataloader.dataset.rescale(torch.cat(outputs), "observation")


    for target, output in zip(targets.T, outputs.T):
        plt.figure()
        plt.plot(target.numpy(), label="observations", lw=3)
        plt.plot(output.numpy(), label="predictions")
        plt.legend()

def compute_occupancy(
    date: datetime, delta: float = 15.0, talon: float = 0.2
) -> np.ndarray:
    if date.weekday() > 4:
        return 0
    if date.hour < 8 or date.hour > 18:
        return 0

    date_start_lockdown = datetime.date(year=2020, month=3, day=17)
    date_end_lockdown = datetime.date(year=2020, month=5, day=11)

    # Full occupancy before lockdown
    occupancy = int(date < date_start_lockdown)

    # After lockdown
    if date_end_lockdown < date:
        # Ocupancy increases gradually
        occupancy += 1 - np.exp(-(date.date() - date_end_lockdown).days / delta)

        # Fixed redduction
        occupancy -= occupancy * talon

    return occupancy
