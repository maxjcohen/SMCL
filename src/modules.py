import torch
import torch.nn as nn

from smcl.smcl import SMCL


class SMCM(nn.Module):
    """Example SMC module building a SMCL on top of a generic input model.

    Define a 3 layer `GRU` input model, along with a `SMCL`. The forward function goes
    through each network, producing `u_tilde` from the input model and `y_hat` from the
    `SMCL`. Uncertainty estimation is delayed to the `SMCL` itself.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, N: int = 100
    ):
        """Define both input model and `SMCL`.

        Parameters
        ----------
        input_size: dimension of the vectors in input sequence.
        hidden_size: defines the dimension of the latent vectors.
        output_size: dimension of the vectors in the output sequence.
        N: number of particle for the `SMCL`. Default is `100`.
        """
        super().__init__()

        self.input_model = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=3, dropout=0.2
        )
        self.smcl = SMCL(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size,
            n_particles=N,
        )

    def forward(self, u: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """Propagates input tensor through the module.

        If `y` is provided, it will be transmitted to the SMCL, resulting in filtering
        computations.
        """
        u_tilde = self.input_model(u)[0]
        y_hat = self.smcl(u_tilde, y)
        return y_hat

    def uncertainty_estimation(self, u, y=None, p=0.05, observation_noise=True):
        u_tilde = self.input_model(u)[0]
        return self.smcl.uncertainty_estimation(
            u_tilde, y=y, p=p, observation_noise=observation_noise
        )
