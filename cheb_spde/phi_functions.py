"""Evaluation of matrix phi-functions via the modified Talbot contour quadrature.

For a matrix A and step h, the phi-functions are defined by

    phi_0(z) = e^z,
    phi_k(z) = integral_0^1 e^{(1-s)z} s^{k-1}/(k-1)! ds, k >= 1.

These satisfy the recurrence phi_{k+1}(z) = (phi_k(z) - 1/k!) / z (for scalar z).
For matrices we evaluate phi_k(hA) via a Cauchy integral on a Talbot contour,
using the parabolic-hyperbolic parameterization of Trefethen--Weideman--Schmelzer
(BIT Numer. Math. 46 (2006) 653--670).

The scalar Kassam--Trefethen evaluation (SIAM J. Sci. Comput. 26 (2005) 1214) is
not used here because L_int for Chebyshev collocation is full and non-normal.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg


# --- Talbot contour parameters (Trefethen--Weideman--Schmelzer 2006) ---
# These are the "optimal-parabola" constants tuned for exp(A*h) contour integration.
_TALBOT_A = 0.5017
_TALBOT_B = 0.6407
_TALBOT_C = 0.6122
_TALBOT_D = 0.2645


def _talbot_path(n_q: int) -> tuple[np.ndarray, np.ndarray]:
    """Return midpoint-rule contour angles theta and psi'(theta)."""
    dth = 2.0 * np.pi / n_q
    theta = -np.pi + (np.arange(n_q) + 0.5) * dth
    psi_prime = _TALBOT_A * (
        1.0 / np.tan(_TALBOT_B * theta)
        - _TALBOT_B * theta / (np.sin(_TALBOT_B * theta) ** 2)
    ) + 1j * _TALBOT_D
    return theta, psi_prime


def _talbot_nodes(theta: np.ndarray) -> np.ndarray:
    """Return psi(theta) -- the Talbot parameterization."""
    return _TALBOT_A * theta / np.tan(_TALBOT_B * theta) - _TALBOT_C + 1j * _TALBOT_D * theta


def phi_k_matrix(k: int, A: np.ndarray, h: float, n_q: int = 64) -> np.ndarray:
    """Compute phi_k(h * A) via the Talbot contour quadrature.

    Parameters
    ----------
    k : int, >= 1
        Order of the phi-function.
    A : (m, m) ndarray
        Dense matrix argument.
    h : float
        Step size (the quadrature is formulated to keep e^{h s} well-scaled).
    n_q : int, default 64
        Number of quadrature nodes. 32--64 is usually sufficient for
        double-precision accuracy on moderate spectra.

    Returns
    -------
    (m, m) ndarray : Re[phi_k(h * A)]
    """
    if k < 1:
        raise ValueError("phi_k_matrix: use scipy.linalg.expm for k = 0")
    m = A.shape[0]
    Id = np.eye(m, dtype=np.complex128)

    theta, psi_p = _talbot_path(n_q)
    psi = _talbot_nodes(theta)
    # s-plane nodes and e^{h s} weights; both s and s' carry the (n_q/h) Jacobian
    s = (n_q / h) * psi
    s_prime = (n_q / h) * psi_p
    e = np.exp(n_q * psi)  # = exp(h * s)

    phi = np.zeros((m, m), dtype=np.complex128)
    for sk, spk, ek in zip(s, s_prime, e):
        # Solve (sk * I - A) X = I via LU, then multiply by weights
        R = scipy.linalg.solve(sk * Id - A, Id)
        phi += ek * (spk / sk**k) * R

    dth = 2.0 * np.pi / n_q
    phi *= dth / (2.0 * np.pi * 1j)
    phi /= h**k  # normalize to phi_k(h * A)
    return np.real(phi)


def all_phi_functions(
    A_int: np.ndarray, h: float, n_q: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the matrix functions needed for one ETDRK4 step.

    Returns
    -------
    (phi0_full, phi1_full, phi2_full, phi3_full, phi0_half, phi1_half)
        phi0_full = exp(h * A_int);   phi0_half = exp((h/2) * A_int)
        phi_k_full = phi_k(h * A_int) for k = 1, 2, 3
        phi1_half  = phi_1((h/2) * A_int)
    """
    phi0_full = scipy.linalg.expm(A_int * h)
    phi0_half = scipy.linalg.expm(A_int * h / 2.0)
    phi1_full = phi_k_matrix(1, A_int, h, n_q)
    phi2_full = phi_k_matrix(2, A_int, h, n_q)
    phi3_full = phi_k_matrix(3, A_int, h, n_q)
    phi1_half = phi_k_matrix(1, A_int, h / 2.0, n_q)
    return phi0_full, phi1_full, phi2_full, phi3_full, phi0_half, phi1_half
