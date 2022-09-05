import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks.progress import ProgressBarBase
import plotly.graph_objects as go
from aim import Figure
from tqdm import tqdm


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    commands: torch.Tensor,
    observations: torch.Tensor,
    hide_start: int = 0,
) -> torch.Tensor:
    """Compute SMC predictions.

    The default behavior of this function is to compute `t+1` predictions, i.e. compute
    the prediction of the SMC model at each time step given the previous one. The
    optional argument `hide_start` can be set to any index in order to hide observation
    from that point, effectively computing prediction for the following time steps with
    any further knowledge.

    Parameters
    ----------
    model:
        Callable implementing the `uncertainty_estimation` function.
    commands:
        Commands tensor with shape `(T, BS, d_in)`.
    observations:
        Full observations with shape `(T, BS, d)`.
    hide_start:
        If any positive index is set, observations are hidden from the model starting at
        that index. The default index `0` does not hide observations at all.

    Returns
    -------
    predictions with shape `(T, BS, N, d)`.
    """
    observations_mask = torch.zeros(observations.shape, dtype=bool)
    if hide_start:
        observations_mask[hide_start:] = True
    netout = model.uncertainty_estimation(
        commands, y=observations.masked_fill(observations_mask, float("nan")), p=0.05
    )
    return netout


def plot_particules_prediction(
    observations: torch.Tensor,
    predictions: torch.Tensor,
    idx_sample: int = 0,
    burnin: int = 0,
):
    """Plot SMC predictions with boxplots.

    At each time step, display a box plot of the generated particules. The `predictions`
    input can be computed with the `predict` function.

    Parameters
    ----------
    observations:
        Full observations with shape `(T, BS, d)`.
    predictions:
        Generated particules with shape `(T, BS, N, d)`.
    idx_sample:
        Index of the sample to plot among the batch. Default is `0`.
    """
    predictions = predictions[burnin:, idx_sample, :].squeeze().numpy()
    observations = observations[burnin:, idx_sample, :].squeeze().numpy()

    plt.boxplot(
        predictions.T,
        positions=np.arange(predictions.shape[0]),
        sym="",
        whis=(0, 100),  # 95% already selected here
        widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor="skyblue"),
    )
    plt.plot(predictions.mean(-1), lw=5, label="SMCL")
    plt.plot(observations, "--", lw=6, label="Observations", zorder=100)
    plt.legend()
    plt.xticks(rotation=45)
    plt.xlabel("time (hour)")
    plt.ylabel("Temperature (°C)")


def plot_range(array, label=None):
    mean = array.mean(axis=-1)
    std = array.std(axis=-1)

    plt.plot(mean, label=label)
    plt.fill_between(np.arange(len(array)), mean - 3 * std, mean + 3 * std, alpha=0.3)


def compute_cost(model, dataloader, loss_function=None):
    loss_function = loss_function or torch.nn.functional.mse_loss
    running_loss = 0
    with torch.no_grad():
        for u, y in dataloader:
            u = u.transpose(0, 1)
            y = y.transpose(0, 1)

            netout = model(u)

            running_loss += loss_function(netout.squeeze(), y.squeeze())
    return running_loss / len(dataloader)


def uncertainty_estimation(model, dataloader):
    accuracy = 0
    for u, y in dataloader:
        u = u.transpose(0, 1)
        y = y.transpose(0, 1)

        with torch.no_grad():
            netout = model(u=u, y=y, noise=True)

        mean = netout * model.W.unsqueeze(-1)
        mean = mean.sum(-2)

        std = netout.square() * model.W.unsqueeze(-1)
        std = std.sum(-2)
        std = std + torch.diag(model.sigma_y2.detach()) - mean.square()

        comparison = ((mean - 3 * std) < y) & (y < (mean + 3 * std))

        accuracy += comparison.to(dtype=float).mean()
    return accuracy / len(dataloader)


class LitProgressBar(ProgressBarBase):
    def __init__(self):
        super().__init__()

        self.global_desc = "Epoch: {epoch}/{max_epoch}"
        self.global_pb = None

    def on_fit_start(self, trainer, pl_module):
        desc = self.global_desc.format(
            epoch=trainer.current_epoch + 1, max_epoch=trainer.max_epochs
        )

        self.global_pb = tqdm(
            desc=desc,
            total=trainer.max_epochs,
            initial=trainer.current_epoch,
            leave=True,
        )

    def on_fit_end(self, trainer, pl_module):
        self.global_pb.close()

    def on_epoch_end(self, trainer, pl_module):

        # Set description
        desc = self.global_desc.format(
            epoch=trainer.current_epoch + 1, max_epoch=trainer.max_epochs
        )
        self.global_pb.set_description(desc)

        # Set logs and metrics
        #         logs = pl_module.logs
        #         for k, v in logs.items():
        #             if isinstance(v, torch.Tensor):
        #                 logs[k] = v.squeeze().item()
        #         self.global_pb.set_postfix(logs)

        # Update progress
        self.global_pb.update(1)


def aim_fig_plot_ts(arrays):
    fig = go.Figure()
    for name, array in arrays.items():
        fig.add_trace(
            go.Scatter(
                y=array.detach().cpu().numpy().squeeze(), mode="lines", name=name
            )
        )
    fig.update_layout(
        width=1000,
        height=450,
        legend=dict(xanchor="left", x=0.5),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    return Figure(fig)


def flatten_batches(array):
    return torch.cat(
        [sample for sample in torch.cat(array, dim=1).transpose(0, 1)], dim=0
    )
