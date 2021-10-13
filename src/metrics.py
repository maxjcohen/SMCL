import torch

# TODO: Add wrapper function as decorator for dataloader iteration


@torch.no_grad()
def compute_cost(model, dataloader, criteria=None):
    criteria = criteria or torch.nn.MSELoss(reduction="none")
    running_loss = []
    for u, y in dataloader:
        u = u.transpose(0, 1)
        y = y.transpose(0, 1)
        netout = model(u)
        running_loss.append(criteria(netout.squeeze(), y.squeeze()))
    running_loss = torch.cat(running_loss, dim=1).mean(0).mean(-1)
    return running_loss


@torch.no_grad()
def pi_metrics(model, dataloader, day_hide=0):
    """Compute PCIP and MPIW.

    Introduced in https://arxiv.org/pdf/1802.07167.pdf
    Prediction Interval Coverage Probability
    Mean Prediction Interval Width
    """
    picp = []
    mpiw = []

    for u, y in dataloader:
        u = u.transpose(0, 1)
        y = y.transpose(0, 1)

        netout = model(u=u, y=y)

        lower_bound = netout.min(dim=2).values
        upper_bound = netout.max(dim=2).values

        picp.append(((lower_bound < y) & (y < upper_bound)).to(dtype=torch.float))
        mpiw.append((upper_bound - lower_bound))

    picp = torch.cat(picp, dim=1).mean(axis=0).mean(axis=-1)
    mpiw = torch.cat(mpiw, dim=1).mean(axis=0).mean(axis=-1)
    return picp, mpiw
