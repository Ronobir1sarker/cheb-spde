"""Chebyshev collocation grid and barycentric differentiation matrices.

Reference: L. N. Trefethen, Spectral Methods in MATLAB (SIAM, 2000), Ch. 6.
"""
from __future__ import annotations

import numpy as np


class ChebyshevGrid:
    """Chebyshev--Gauss--Lobatto grid (points of the second kind) on [xmin, xmax].

    Attributes
    ----------
    x : (N+1,) ndarray
        Collocation nodes, ordered from x[0] = xmax down to x[N] = xmin
        (reversed relative to the usual left-to-right convention, because
        the cosine map is monotonically decreasing).
    w : (N+1,) ndarray
        Barycentric interpolation weights.
    D : (N+1, N+1) ndarray
        First-derivative differentiation matrix.
    D2 : (N+1, N+1) ndarray
        Second-derivative differentiation matrix, D2 = D @ D.
    """

    def __init__(self, N: int, xmin: float = 0.0, xmax: float = 1.0) -> None:
        if N < 1:
            raise ValueError("N must be >= 1")
        self.N = N
        self.xmin = xmin
        self.xmax = xmax
        self.x, self.w = self._build_nodes_and_weights()
        self.D = self._build_differentiation_matrix()
        self.D2 = self.D @ self.D

    def _build_nodes_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        N = self.N
        k = np.arange(N + 1)
        # Chebyshev points of the second kind, mapped linearly to [xmin, xmax]
        x = 0.5 * (self.xmin + self.xmax) + 0.5 * (self.xmax - self.xmin) * np.cos(
            k * np.pi / N
        )
        # Barycentric weights for the second-kind grid
        w = (-1.0) ** k
        w[0] *= 0.5
        w[-1] *= 0.5
        return x, w

    def _build_differentiation_matrix(self) -> np.ndarray:
        N = self.N
        x, w = self.x, self.w
        D = np.zeros((N + 1, N + 1))
        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    D[i, j] = (w[j] / w[i]) / (x[i] - x[j])
        # Diagonal entries fixed by "negative sum trick"
        for i in range(N + 1):
            D[i, i] = -np.sum(D[i, :])
        return D

    def interpolate(self, f_values: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        """Barycentric interpolation of function values at new points.

        Parameters
        ----------
        f_values : (N+1,) ndarray
            Function values at self.x.
        x_eval : (M,) ndarray
            Evaluation points.

        Returns
        -------
        (M,) ndarray of interpolated values.
        """
        x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
        num = np.zeros_like(x_eval)
        den = np.zeros_like(x_eval)
        for j in range(self.N + 1):
            # guard against exact hits: set a large temp to dominate
            diff = x_eval - self.x[j]
            # avoid division-by-zero with np.errstate; we patch exact hits below
            with np.errstate(divide="ignore", invalid="ignore"):
                temp = self.w[j] / diff
            num += temp * f_values[j]
            den += temp
        out = num / den
        # Fix any exact hits
        for i in range(len(x_eval)):
            hits = np.where(np.isclose(x_eval[i], self.x))[0]
            if len(hits) > 0:
                out[i] = f_values[hits[0]]
        return out
