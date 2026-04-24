"""Chebyshev exponential-integrator SPDE solver on a bounded interval.

Solves SPDEs of the form

    du = (L u + N(u)) dt  +  dW_noise,     x in [a, b],  u(a,t) = gL, u(b,t) = gR,

where:
    L       = second-derivative (Laplacian) operator on the Chebyshev grid
    N(u)    = user-supplied nonlinearity (e.g. Allen-Cahn: N(u) = u - u^3)
    dW_noise is either additive space-time white noise with intensity sigma,
             or an OU-in-time, white-in-space field eta(x,t).

METHOD
------
Integrator: "stochastic exponential Euler" in the sense of Lord-Tambue /
Jentzen-Kloeden. Writing u = v + ell where ell is the linear boundary lift,
the interior variable v satisfies homogeneous Dirichlet BCs and

    dv = (L_int v + F(v, t)) dt + dW_int(t),

with F absorbing the lifted nonlinearity and boundary column terms (exactly
as in the deterministic gBH solver). One step of size h is

    v_{n+1} = phi0 * v_n  +  h * phi1 * F(v_n, t_n)  +  xi_n,

    phi0   = exp(h * L_int),
    phi1   = phi_1(h * L_int)  (evaluated by Talbot contour on the dense matrix)

    xi_n ~ N(0, C),      C = integral_0^h exp(s L_int) S exp(s L_int^T) ds,

where S = diag(sigma_i^2 / w_i) is the covariance of the spatial noise
(white or OU-projected) and w_i are the Clenshaw-Curtis weights.

For the stochastic convolution we use the fact that L_int is diagonalisable
as L_int = V Lambda V^{-1}. In the eigenbasis the covariance C becomes

    C_tilde_{ij} = S_tilde_{ij} * (exp((lam_i + lam_j) h) - 1) / (lam_i + lam_j),

and the noise increment is sampled by Cholesky of C_tilde (a small dense
matrix of size (N-1) x (N-1)) and mapped back to physical space via V. This
is O(N^3) once per (L_int, h) pair at setup, then O(N^2) per step -- the
same cost class as the deterministic Cheb-ETDRK4 solver.

For OU noise eta(x,t), we treat eta as a *separate* process integrated by
its own exact discretisation (src/noise.py), and pass sigma * eta(t_n) as
an additive drift at the start of each step. This is weak order 1 in dt.

References
----------
[1] Lord & Tambue, IMA J. Numer. Anal. 33 (2013) 515-543
    (exponential integrators for parabolic SPDEs with additive noise).
[2] Jentzen & Kloeden, Proc. R. Soc. A 465 (2009) 649-667
    (rate one approximation for parabolic SPDE).
[3] Trefethen, Weideman, Schmelzer, BIT Numer. Math. 46 (2006) 653-670
    (Talbot contour for matrix phi-functions).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import scipy.linalg

from .chebyshev_grid import ChebyshevGrid
from .phi_functions import phi_k_matrix
from .noise import clenshaw_curtis_weights, OUNoiseCheb


class ChebSPDESolver:
    """Exponential-Euler SPDE solver on a Chebyshev grid with Dirichlet BCs.

    Parameters
    ----------
    N : int
        Number of Chebyshev intervals (N+1 nodes, N-1 interior).
    xmin, xmax : float
        Spatial domain.
    nonlinearity : callable (u_int, t) -> (N-1,) array
        Pure reaction part N(u) evaluated at interior nodes.
    gL, gR : callable t -> float
        Time-dependent Dirichlet boundary data.
    noise_type : {"white", "ou", "none"}
        Additive noise model.
    sigma : float
        Noise intensity (ignored if noise_type == "none").
    tau : float, optional
        OU correlation time (required if noise_type == "ou").
    initial_condition : callable (x,) -> (N+1,)
        u0(x) on the full grid.
    n_quad : int
        Number of Talbot quadrature nodes for phi-functions.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        N: int,
        xmin: float,
        xmax: float,
        nonlinearity: Callable,
        gL: Callable,
        gR: Callable,
        noise_type: str = "none",
        sigma: float = 0.0,
        tau: Optional[float] = None,
        initial_condition: Optional[Callable] = None,
        n_quad: int = 64,
        seed: int = 0,
    ):
        if noise_type not in ("white", "ou", "none"):
            raise ValueError("noise_type must be 'white', 'ou', or 'none'")
        if noise_type == "ou" and (tau is None or tau <= 0):
            raise ValueError("OU noise requires a positive tau")
        self.N = N
        self.grid = ChebyshevGrid(N, xmin, xmax)
        self.nonlinearity = nonlinearity
        self.gL = gL
        self.gR = gR
        self.noise_type = noise_type
        self.sigma = sigma
        self.tau = tau
        self.u0_fn = initial_condition
        self.n_quad = n_quad
        self.rng = np.random.default_rng(seed)

        # Clenshaw-Curtis weights on the full grid, then restrict to interior
        self.weights = clenshaw_curtis_weights(N, xmin, xmax)
        self.w_int = self.weights[1:-1]  # (N-1,)

        # Second-derivative operator on interior
        self.L_int = self.grid.D2[1:-1, 1:-1].copy()

        # Boundary-column pieces (for the lifted nonlinearity)
        D2 = self.grid.D2
        self.L_col_0 = D2[1:-1, 0].copy()
        self.L_col_N = D2[1:-1, -1].copy()

        # Precomputed stage caches (populated by prepare_step)
        self._h = None
        self._phi0 = None
        self._phi1 = None
        self._chol_C = None   # Cholesky factor for the stochastic convolution

        # OU process if requested
        self._ou = None

    # ---- Boundary lifting ----
    def _ell(self, x: np.ndarray, t: float) -> np.ndarray:
        a, b = self.grid.xmin, self.grid.xmax
        return ((b - x) / (b - a)) * self.gL(t) + ((x - a) / (b - a)) * self.gR(t)

    def _ell_t(self, x: np.ndarray, t: float) -> np.ndarray:
        """Complex-step time derivative of ell."""
        a, b = self.grid.xmin, self.grid.xmax
        eps = 1e-20
        gL_t = np.imag(self.gL(t + 1j * eps)) / eps
        gR_t = np.imag(self.gR(t + 1j * eps)) / eps
        return ((b - x) / (b - a)) * gL_t + ((x - a) / (b - a)) * gR_t

    # ---- Setup (cached for fixed dt) ----
    def prepare_step(self, h: float) -> None:
        """Precompute phi_0, phi_1 and the stochastic-convolution factor."""
        if self._h == h and self._phi0 is not None:
            return
        self._h = h
        self._phi0 = scipy.linalg.expm(self.L_int * h)
        self._phi1 = phi_k_matrix(1, self.L_int, h, n_q=self.n_quad)

        if self.noise_type == "white":
            self._build_white_noise_factor(h)
        elif self.noise_type == "ou":
            # OU provides its own exact spatial covariance; the stochastic
            # convolution here is just the exponential Euler convolution of
            # a deterministic drift, so no extra factor is needed.
            self._chol_C = None
            self._ou = OUNoiseCheb(
                N=self.N, weights=self.weights, tau=self.tau,
                sigma=self.sigma, dt=h, rng=self.rng,
            )

    def _build_white_noise_factor(self, h: float) -> None:
        """Compute Cholesky factor of the stochastic convolution covariance.

        For dv = L_int v dt + sigma diag(1/sqrt(w_int)) dW,  the exact
        covariance of the Ito integral over one step is

            C = sigma^2 * int_0^h exp(s L_int) W^{-1} exp(s L_int^T) ds,

        where W = diag(w_int) is the Clenshaw-Curtis weight matrix.

        We diagonalise L_int = V Lambda V^{-1} once and evaluate C in the
        eigenbasis (closed form), then take Cholesky and map back. For
        non-normal L_int, V is complex; we work with complex arithmetic
        throughout, then take the real part of the final factor (the
        covariance is real by construction).
        """
        sigma = self.sigma
        if sigma == 0.0:
            self._chol_C = None
            return

        # Diagonalise L_int (dense, small: (N-1) x (N-1))
        Lam, V = scipy.linalg.eig(self.L_int)      # V @ diag(Lam) @ inv(V) = L_int
        Vinv = scipy.linalg.inv(V)

        # Spatial noise covariance in physical space: sigma^2 / w_i on diagonal
        W_inv = np.diag(1.0 / self.w_int)
        S_tilde = Vinv @ W_inv @ Vinv.conj().T

        # C_tilde_{ij} = S_tilde_{ij} * (exp((lam_i + lam_j) h) - 1) / (lam_i + lam_j)
        m = self.L_int.shape[0]
        LL = Lam[:, None] + Lam.conj()[None, :]
        # careful with small LL
        small = np.abs(LL) < 1e-14
        factor = np.where(
            small,
            h,  # limit as LL -> 0
            (np.exp(LL * h) - 1.0) / np.where(small, 1.0, LL),
        )
        C_tilde = sigma**2 * S_tilde * factor

        # Back to physical space
        C_phys = V @ C_tilde @ V.conj().T
        # Should be real, symmetric, PSD; symmetrize and take real
        C_phys = 0.5 * (C_phys + C_phys.conj().T)
        C_phys_real = np.real(C_phys)

        # Cholesky (with a tiny jitter if needed for the smallest eigenvalues)
        jitter = 0.0
        for attempt in range(6):
            try:
                L_chol = np.linalg.cholesky(
                    C_phys_real + jitter * np.eye(m)
                )
                self._chol_C = L_chol
                return
            except np.linalg.LinAlgError:
                jitter = max(jitter * 10.0, 1e-14 * np.trace(C_phys_real) / m)
        raise RuntimeError(
            "Stochastic-convolution covariance failed Cholesky. "
            "Check sigma, h, or N."
        )

    # ---- F-tilde: lifted nonlinearity on interior ----
    def _F_tilde(self, v_int: np.ndarray, t: float) -> np.ndarray:
        x_int = self.grid.x[1:-1]
        ell_int = self._ell(x_int, t)
        ell_t_int = self._ell_t(x_int, t)
        u_int = v_int + ell_int
        # N(u) at interior nodes (user-supplied)
        Nu = self.nonlinearity(u_int, t)
        # (L * ell)|_int reconstructed from boundary columns
        L_ell_int = (
            self.grid.D2[1:-1, 1:-1] @ ell_int
            + self.L_col_0 * self.gR(t)
            + self.L_col_N * self.gL(t)
        )
        return Nu - ell_t_int + L_ell_int

    # ---- Time loop ----
    def solve(
        self,
        T: float,
        dt: float,
        store_every: int = 1,
        progress: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance from t = 0 to t = T with step dt.

        Returns
        -------
        u_hist : (n_store, N+1)
        t_hist : (n_store,)
        """
        self.prepare_step(dt)
        x = self.grid.x
        n_steps = int(round(T / dt))
        n_store = n_steps // store_every + 1

        u = np.zeros((n_store, self.N + 1))
        t_hist = np.zeros(n_store)

        # Initial condition
        if self.u0_fn is None:
            u0_vec = np.zeros(self.N + 1)
        else:
            u0_vec = self.u0_fn(x)
        u[0] = u0_vec
        t_hist[0] = 0.0

        v_int = u0_vec[1:-1] - self._ell(x[1:-1], 0.0)

        store_idx = 1
        for n in range(n_steps):
            t_n = n * dt
            # Deterministic exponential-Euler step
            F_n = self._F_tilde(v_int, t_n)
            # OU noise adds as an additive drift at interior nodes (weak order 1)
            if self.noise_type == "ou":
                eta = self._ou.step()
                F_n = F_n + eta[1:-1]
            v_new = self._phi0 @ v_int + dt * (self._phi1 @ F_n)

            # Stochastic convolution (additive white noise)
            if self.noise_type == "white" and self._chol_C is not None:
                z = self.rng.standard_normal(v_int.shape[0])
                v_new = v_new + self._chol_C @ z

            v_int = v_new
            t_np1 = (n + 1) * dt

            if (n + 1) % store_every == 0 and store_idx < n_store:
                u_full = np.zeros(self.N + 1)
                u_full[1:-1] = v_int + self._ell(x[1:-1], t_np1)
                u_full[0] = self.gR(t_np1)   # x = xmax node (by CGL ordering)
                u_full[-1] = self.gL(t_np1)  # x = xmin
                u[store_idx] = u_full
                t_hist[store_idx] = t_np1
                store_idx += 1

            if progress and (n + 1) % max(1, n_steps // 20) == 0:
                print(f"  step {n+1}/{n_steps}  ({100*(n+1)/n_steps:.0f}%)")

        return u[:store_idx], t_hist[:store_idx]
