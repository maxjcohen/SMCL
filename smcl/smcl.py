import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

# TODO: swich for range with joint iteration over u and y


class SMCL(nn.RNN):
    """Sequential Monte Carlo Layer.

    This hybrid layer enables computing particle filter based on an RNN architecture and
    weights. For traditional inference and learning, the forward function falls back to
    the inherited PyTorch RNN function, followed by a Linear application, for maximum
    compatibility and speed.

    In this document, we refer to dimensions as:

     - `BATCH_SIZE`: batch size.
     - `D_IN`: input size.
     - `D_HIDDEN`: hidden dimension.
     - `D_OUT`: output dimension.
     - `T`: number of time steps.
     - `N`: number of particles.

    Arguments
    ---------
    input_size: dimension of input vectors at each time step (`D_IN`).
    hidden_size: dimension of the state space vectors (`D_HIDDEN`)
    output_size: dimension of observations samples at each time step (`D_OUT`).
    n_particles: number of particles when computing particle filter (`N`).
    Default `100`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_particles: int = 100,
        **kwargs
    ):
        super().__init__(input_size=input_size, hidden_size=hidden_size, **kwargs)

        assert self.num_layers == 1, "SMCL is restricted to a single layer."
        assert self.batch_first == False, "SMCL isn't implemented for batch first."
        assert self.dropout == 0, "SMCL is restricted to zero dropout"
        assert self.bidirectional == False, "SMCL restricted to unidirectional."

        self._output_size = output_size
        self._n_particles = n_particles

        self._g = nn.Linear(self.hidden_size, self._output_size)

        self._sigma_x = nn.Parameter(
            torch.randn(self.hidden_size).abs(), requires_grad=False
        )
        self._sigma_y = nn.Parameter(
            torch.randn(self._output_size).abs(), requires_grad=False
        )

        self.softmax = nn.Softmax(dim=1)

    @property
    def sigma_x2(self) -> torch.Tensor:
        return torch.diag(self._sigma_x)

    @sigma_x2.setter
    def sigma_x2(self, matrix: torch.Tensor):
        self._sigma_x.data = torch.diag(matrix)

    @property
    def sigma_y2(self) -> torch.Tensor:
        return torch.diag(self._sigma_y)

    @sigma_y2.setter
    def sigma_y2(self, matrix: torch.Tensor):
        self._sigma_y.data = torch.diag(matrix)

    def _f(
        self, input_vector: torch.Tensor, hidden_vector: torch.Tensor
    ) -> torch.Tensor:
        """Modified state space model.

        Compute the same RNNCell function as in PyTorch, without the activation
        function.

        Arguments
        ---------
        input_tensor: input vector with dimension `(BATCH_SIZE, N, D_IN)`.
        hidden_tensor: hidden vector with dimension `(BATCH_SIZE, N, D_HIDDEN)`.

        Returns
        -------
        Output tensor with dimension `(BATCH_SIZE, N, D_HIDDEN)`
        """
        input_vector = F.linear(
            input_vector, weight=self.weight_ih_l0, bias=self.bias_ih_l0
        )
        hidden_vector = F.linear(
            hidden_vector, weight=self.weight_hh_l0, bias=self.bias_hh_l0
        )
        input_vector = input_vector.unsqueeze(-2)
        output = input_vector + hidden_vector
        return output

    def forward(self, u: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """Compute inference as traditional RNN or particle filter.

        If the observation tensor `y` is not provided, the function falls back to the
        inherited PyTorch forward function. Otherwise, we compute the particle filter
        given the observations. Missing observations can be provided with value
        `float('nan')`.

        Arguments
        ---------
        u: input tensor with shape `(T, BATCH_SIZE, D_IN)`.
        y: observation tensor with shape `(T, BATCH_SIZE, D_OUT)`.

        Returns
        -------
        Predictions tensor with shape `(T, BATCH_SIZE, N, D_OUT)`. In case observations
        are not provided, `N` is set to `1` and the third dimension is squeezed.
        """
        if y is None:
            netout = super().forward(u)[0]
            netout = self._g(netout)
            return netout
        self._u = u

        T = u.shape[0]
        bs = u.shape[1]

        predictions = []
        self._particules = []
        self._I = []
        self._W = []

        # Generate initial particles
        x = torch.zeros(bs, self.N, self.hidden_size, device=u.device)
        self._eta = MultivariateNormal(
            loc=torch.zeros(x.shape, device=self.sigma_x2.device),
            covariance_matrix=self.sigma_x2,
        )

        # Iterate k through time
        for k in range(T):
            if k > 0 and not y[k - 1].isnan().any():
                self.w = self.compute_weights(
                    y[k - 1], predictions[k - 1].transpose(0, 1)
                )
                self._W.append(self.w)

                # Select sampled indices from previous time step
                I = torch.multinomial(self.w, self.N, replacement=True)
                self._I.append(I)
                x = self.__class__.select_indices(x, I)

            # Compute new hidden state
            x = torch.tanh(self._f(u[k], x) + self._eta.sample())
            self._particules.append(x)

            # Compute new weights
            y_hat = self._g(x)
            predictions.append(y_hat)

        self.w = self.compute_weights(y[-1], predictions[-1].transpose(0, 1))
        self._W.append(self.w)

        return torch.stack(predictions)

    def compute_weights(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Compute importance weights for predictions.

        Compute weights associated with particles' predictions.

        Arguments
        ---------
        y: observations with shape `(BATCH_SIZE, D_OUT)`.
        y_hat: predictions, associated with particles, with shape `(BATCH_SIZE, N,
        D_OUT)`.

        Returns
        -------
        Weights with shape `(BATCH_SIZE, N)`.
        """
        _normal_y = MultivariateNormal(y, covariance_matrix=self.sigma_y2)
        w = _normal_y.log_prob(y_hat).T
        w = self.softmax(w)
        return w

    @staticmethod
    def select_indices(x, I):
        return torch.cat(
            [x[batch_idx, particule_idx] for batch_idx, particule_idx in enumerate(I)]
        ).view(x.shape)

    def smooth_pms(self, x, I):
        N = I.shape[-1]
        T = x.shape[0]
        bs = x.shape[1]

        # Initialize flat indexing
        I_flat = torch.zeros((T, bs, N), dtype=torch.long)
        I_flat[-1] = torch.arange(N)

        # Fill flat indexing with reversed indexing
        for k in reversed(range(T - 1)):
            I_flat[k] = self.__class__.select_indices(I[k], I_flat[k + 1])

        self._I_flat = I_flat
        # Stack all selected particles
        return torch.stack(
            [self.__class__.select_indices(x_i, I_i) for x_i, I_i in zip(x, I_flat)]
        )

    def uncertainty_estimation(self, u, y, p=0.05, observation_noise=True):
        predictions = self(u, y)
        if observation_noise:
            _normal_y = MultivariateNormal(predictions, covariance_matrix=self.sigma_y2)
            predictions = _normal_y.sample()
        return predictions[:, :, int(p * self.N) : -int(p * self.N)]

    def compute_cost(self, y):
        # Smooth
        particules = self.smooth_pms(self.particules, self.I).detach()

        # Compute likelihood
        normal_y = MultivariateNormal(y.unsqueeze(-2), covariance_matrix=self.sigma_y2)
        loss_y = -normal_y.log_prob(self._g(particules)) * self.w.detach()
        loss_y = loss_y.sum(-1)

        normal_x = MultivariateNormal(
            torch.atanh(particules[1:]), covariance_matrix=self.sigma_x2
        )
        loss_x = -normal_x.log_prob(self._f(self._u[1:], particules[:-1]))
        # Log of d arctanh(x)
        ldx = torch.log(
            1 - particules[1:].square() + torch.finfo(torch.float32).eps
        ).mean(-1)
        loss_x = loss_x + ldx
        loss_x = loss_x * self.w.detach()
        loss_x = loss_x.sum(-1)

        # Aggregate terms
        return loss_x.mean() + loss_y.mean()

    def compute_sigma_y(self, y):
        # Smooth
        particules = self.smooth_pms(self.particules, self.I).detach()

        sigma_y2 = y.unsqueeze(-2) - self._g(particules)

        # Compute square
        sigma_y2 = sigma_y2.square()

        # Sum on time steps
        sigma_y2 = sigma_y2.mean(0)

        # Sum on particules
        sigma_y2 = sigma_y2 * self.w.unsqueeze(-1).detach()
        sigma_y2 = sigma_y2.sum(axis=1)

        # Average on batches
        sigma_y2 = sigma_y2.mean(0)

        return sigma_y2

    def compute_sigma_x(self):
        # Smooth
        particules = self.smooth_pms(self.particules, self.I).detach()

        sigma_x2 = torch.atanh(particules[1:]) - self._f(self._u[1:], particules[:-1])

        # Compute square
        sigma_x2 = sigma_x2.square()

        # Sum on time steps
        sigma_x2 = sigma_x2.mean(0)

        # Sum on particules
        sigma_x2 = sigma_x2 * self.w.unsqueeze(-1).detach()
        sigma_x2 = sigma_x2.sum(axis=1)

        # Average on batches
        sigma_x2 = sigma_x2.mean(0)
        return torch.diag(sigma_x2)

    @property
    def N(self):
        return self._n_particles

    @N.setter
    def N(self, n):
        raise NotImplementedError
        self._n_particles = n

    @property
    def I(self):
        return torch.stack(self._I)

    @property
    def W(self):
        return torch.stack(self._W)

    @property
    def particules(self):
        return torch.stack(self._particules)
