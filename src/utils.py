import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks.progress import ProgressBarBase
from tqdm import tqdm

@torch.no_grad()
def boxplotprediction(model, dataloader_val, u, y, mc_dropout=False, burnin=0, idx_out=0, day_hide=0):
    # Set y mask for SMC
    y_mask = torch.zeros(y.shape, dtype=bool)
    if day_hide:
        y_mask[24*day_hide:] = True
    netout = model.uncertainty_estimation(u, y=y.masked_fill(y_mask, float('nan')), p=0.05)

    netout = dataloader_val.dataset.rescale(netout, "observation")[burnin:, 0, :, idx_out].numpy()
    y = dataloader_val.dataset.rescale(y, "observation").numpy()[burnin:, 0, idx_out]
    return netout, y

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
