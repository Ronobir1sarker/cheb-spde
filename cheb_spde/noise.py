"""Space-time white-noise generation on a Chebyshev-Gauss-Lobatto grid.

The correct discrete approximation of a space-time white noise xi(x,t)
on a non-uniform spatial grid {x_i} is

    dW_i ~ N(0, dt / w_i),

where w_i is the integration weight at node x_i (so that sum_i w_i = b - a).
For a uniform grid w_i = dx and we recover the standard finite-difference
scaling dW_i ~ N(0, dt/dx).

For Chebyshev-Gauss-Lobatto nodes on [a,b], the appropriate w_i are the
Clenshaw-Curtis quadrature weights, computed below via the standard DCT
formulation (Waldvogel 2006, BIT Numer. Math. 46:195-202).

For OU (temporally colored) noise, we use the exact mean-square update

    eta_{n+1} = alpha * eta_n + std * z_n,   z_n ~ N(0, I/W),

with alpha = exp(-dt/tau), std = sigma * sqrt((1 - alpha^2)),
and W = diag(w_i) so that the stationary variance of eta at node i
is sigma^2 / w_i -- the spatial-white-noise limit.
"""
from __future__ import annotations

import numpy as np


def clenshaw_curtis_weights(N: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Clenshaw-Curtis quadrature weights on the CGL grid with N+1 nodes.

    Nodes are x_k = (a+b)/2 + (b-a)/2 * cos(k*pi/N), k = 0,...,N
    (same ordering as ChebyshevGrid: x[0] = b, x[N] = a).

    Implementation: the weights w_k on [-1,1] admit the closed form via
    the DCT of a specific sequence (Waldvogel 2006). We then rescale by
    (b-a)/2 to the physical interval.

    Returns
    -------
    w : (N+1,) ndarray, strictly positive, summing to (b - a).
    """
    if N < 1:
        raise ValueError("N must be >= 1")

    # Weights on [-1, 1] via Waldvogel's DCT-based formula.
    c = np.zeros(N + 1)
    c[0] = 2.0
    c[1::2] = 0.0
    # c_{2k} = 2 / (1 - (2k)^2)  for k >= 1 when 2k <= N
    for k in range(1, N // 2 + 1):
        c[2 * k] = 2.0 / (1.0 - (2 * k) ** 2)
    # End-correction when N is even: halve the Nyquist term
    if N % 2 == 0:
        c[N] = 1.0 / (1.0 - N ** 2)

    # Real DCT-I of c; the output gives the weights directly up to scaling.
    # scipy has this but I'll do it manually with numpy to keep deps minimal.
    w = np.zeros(N + 1)
    k = np.arange(N + 1)
    for n in range(N + 1):
        w[n] = c[0] / 2.0
        for m in range(1, N):
            w[n] += c[m] * np.cos(m * n * np.pi / N)
        w[n] += c[N] / 2.0 * np.cos(N * n * np.pi / N)
        w[n] /= N

    # Boundary nodes (k = 0 and k = N) need halving by the DCT-I convention
    w[0] /= 2.0
    w[-1] /= 2.0
    # Multiply by 2 because DCT-I above is on [0,pi] whereas Clenshaw-Curtis
    # integrates cos(m*theta) weighted against 1 on [0, pi]
    w *= 2.0

    # Rescale to [a, b]
    w = w * (b - a) / 2.0
    return w


def chebyshev_noise_increment(
    rng: np.random.Generator, weights: np.ndarray, dt: float
) -> np.ndarray:
    """Draw one space-time white-noise increment on the Chebyshev grid.

    Returns an (N+1,)-vector dW such that

        Cov(dW_i, dW_j) = (dt / w_i) * delta_{ij}.

    In the limit N -> infinity this approximates

        <dW(x) dW(x')> = dt * delta(x - x').
    """
    return np.sqrt(dt / weights) * rng.standard_normal(weights.shape)


class OUNoiseCheb:
    """Ornstein-Uhlenbeck noise field on a Chebyshev grid.

    The stochastic ODE for the noise field eta(x, t) at each node is

        d eta_i = -(1/tau) eta_i dt + (sigma * sqrt(2/tau)) dW_i,

    where dW_i has the Chebyshev scaling above. We use the exact
    mean-square discretization

        eta_{n+1,i} = alpha * eta_{n,i} + std_i * z,   z ~ N(0,1),

    with alpha = exp(-dt/tau) and std_i = sigma * sqrt((1 - alpha^2) / w_i).
    This reproduces the continuum stationary variance sigma^2 / w_i at each
    node and the continuum autocorrelation exp(-|dt|/tau) exactly.
    """

    def __init__(
        self,
        N: int,
        weights: np.ndarray,
        tau: float,
        sigma: float,
        dt: float,
        rng: np.random.Generator | None = None,
    ):
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.N = N
        self.weights = weights
        self.tau = tau
        self.sigma = sigma
        self.dt = dt
        self.alpha = float(np.exp(-dt / tau))
        self.std = sigma * np.sqrt((1.0 - self.alpha ** 2) / weights)
        self.rng = rng if rng is not None else np.random.default_rng()
        # Initialize at stationary distribution
        self.eta = (sigma / np.sqrt(weights)) * self.rng.standard_normal(N + 1)

    def step(self) -> np.ndarray:
        """Advance the OU field one step of size dt and return the new eta."""
        z = self.rng.standard_normal(self.N + 1)
        self.eta = self.alpha * self.eta + self.std * z
        return self.eta
