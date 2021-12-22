from typing import Union

import torch

def criteria_MSE(model, u, y):
    return torch.nn.functional.mse_loss(model(u), y, reduction="none")


def criteria_PICP(model, u, y):
    netout = model(u, y)
    lower_bound = netout.min(dim=2).values
    upper_bound = netout.max(dim=2).values
    return ((lower_bound < y) & (y < upper_bound)).to(dtype=torch.float32)


def criteria_MPIW(model, u, y):
    netout = model(u, y)
    lower_bound = netout.min(dim=2).values
    upper_bound = netout.max(dim=2).values
    return upper_bound - lower_bound


@torch.no_grad()
def compute_cost(
    model: callable,
    dataloader: torch.utils.data.DataLoader,
    criteria: Union[callable, str] = "mse",
    reduction: str = "none",
) -> torch.Tensor:
    """Compute given cost for the model over the given dataloader.

    Iterate over the dataloader, computing the given cost (default to MSE). Results can
    be averaged by setting the `reduction` parameter.

    Parameters
    ----------
    model: model to evaluate.
    dataloader: iterable over the dataset.
    criteria: cost function. If set to `None`, defaults to `torch.nn.MSELoss`.
    reduction: if set to `"mean"`, average the computed cost over the dataset samples.
    Default is `"none"`.

    Returns
    -------
    Cost tensor with shape `(n_samples x T)` if reduction is `"none"`, scalar if `"mean"`.
    """
    if isinstance(criteria, str):
        try:
            criteria = {
                "mse": criteria_MSE,
                "picp": criteria_PICP,
                "mpiw": criteria_MPIW,
            }[criteria]
        except KeyError:
            raise NameError(f"Critera {criteria} unkown.")
    cost = []
    for u, y in dataloader:
        cost.append(criteria(model, u, y))
    # Concatenate running cost on batch dimension
    cost = torch.cat(cost, dim=1)
    # Average results on feature dimension
    cost = cost.mean(dim=-1)
    # Flatten batch and time dimension
    # cost = cost.flatten(end_dim=1)
    if reduction == "mean":
        cost = cost.mean()
    return cost


def cumulative_cost(
    model: callable,
    dataloader: torch.utils.data.DataLoader,
    eval_time: int = 48,
    delta: int = 48,
    criteria: callable = "mse",  # Or str
    reduction: str = "none",
):
    """Compute cost for t+1 prediction to t+delta.

    Parameters
    ----------
    model: model to evaluate.
    dataloader: iterable over the dataset.
    eval_time: hide observations from this time.
    delta: total number of timesteps without observations to aggregate.
    criteria: cost function. If set to `None`, defaults to `torch.nn.MSELoss`.
    reduction: if set to `"mean"`, average the computed cost over the dataset samples.
    Default is `"none"`.

    Returns
    -------
    Cost tensor with shape `(T, n_samples)` if reduction is `"none"`, `(T)` if `"mean"`.
    """
    # Define model with hidden observations
    def model_hidden(u, y=None):
        if y is not None:
            y_mask = torch.zeros(y.shape, dtype=bool)
            y_mask[eval_time : eval_time + delta] = True
            y = y.masked_fill(y_mask, float("nan"))
            return model(u, y)
        return model(u)

    cost = compute_cost(
        model=model_hidden, dataloader=dataloader, criteria=criteria, reduction="none"
    )  # .reshape(dataloader.dataset.T, -1)
    # Select study time
    cost = cost[eval_time : eval_time + delta]
    # Compute cumulative average
    cost = cost.cumsum(dim=0) / torch.arange(1, delta + 1).unsqueeze(-1)
    if reduction == "mean":
        cost = cost.mean(-1)
    return cost
