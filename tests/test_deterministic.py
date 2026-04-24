"""Test 1: Deterministic limit (sigma = 0).

Heat equation u_t = u_xx on [-1, 1] with homogeneous Dirichlet BCs
and Fourier mode initial condition u0(x) = sin(pi * (x+1)/2) which is
an eigenfunction of the Laplacian with eigenvalue -(pi/2)^2.

Exact solution: u(x, t) = exp(-(pi/2)^2 t) * sin(pi * (x+1)/2).

Our solver should reproduce this to near machine precision for any
reasonable N and dt, because:
  1. sigma = 0 turns off all stochastic terms
  2. nonlinearity = 0 makes F_tilde purely linear
  3. boundary data is zero so ell = 0
  -> the update reduces to v_new = phi0 @ v = exp(h L_int) @ v,
     which is the exact linear propagator on the interior.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from cheb_spde import ChebSPDESolver


def test_deterministic_heat():
    N = 32
    xmin, xmax = -1.0, 1.0
    T = 0.5
    dt = 0.05

    def u0(x):
        return np.sin(np.pi * (x - xmin) / (xmax - xmin))

    def nl(u_int, t):
        return np.zeros_like(u_int)

    gL = lambda t: 0.0 + 0.0 * t  # must accept complex t
    gR = lambda t: 0.0 + 0.0 * t

    solver = ChebSPDESolver(
        N=N, xmin=xmin, xmax=xmax,
        nonlinearity=nl, gL=gL, gR=gR,
        noise_type="none", sigma=0.0,
        initial_condition=u0, seed=0,
    )
    u_hist, t_hist = solver.solve(T=T, dt=dt, store_every=1)

    # Exact solution
    x = solver.grid.x
    lam = -(np.pi / (xmax - xmin)) ** 2 * np.pi  # = -(pi/2)^2 * 4 / (xmax-xmin)^2 ... recompute
    # on [-1, 1]: u0 = sin(pi * (x+1)/2), eigenvalue of d^2/dx^2 is -(pi/2)^2
    lam = -(np.pi / 2.0) ** 2
    u_exact = np.exp(lam * t_hist[-1]) * np.sin(np.pi * (x - xmin) / (xmax - xmin))

    err = np.max(np.abs(u_hist[-1] - u_exact))
    print(f"  N={N}, dt={dt}, T={T}: L_inf error vs exact = {err:.3e}")
    assert err < 1e-12, f"Deterministic heat failed: err = {err:.3e}"
    return err


def test_deterministic_convergence_N():
    """Increase N and watch error drop to machine precision."""
    xmin, xmax = -1.0, 1.0
    T = 0.3
    dt = 0.05
    lam = -(np.pi / 2.0) ** 2

    def u0(x):
        return np.sin(np.pi * (x - xmin) / (xmax - xmin))

    def nl(u_int, t):
        return np.zeros_like(u_int)

    gL = lambda t: 0.0 + 0.0 * t
    gR = lambda t: 0.0 + 0.0 * t

    errs = []
    Ns = [8, 12, 16, 24, 32, 48]
    for N in Ns:
        solver = ChebSPDESolver(
            N=N, xmin=xmin, xmax=xmax,
            nonlinearity=nl, gL=gL, gR=gR,
            noise_type="none", sigma=0.0,
            initial_condition=u0, seed=0,
        )
        u_hist, t_hist = solver.solve(T=T, dt=dt, store_every=1)
        x = solver.grid.x
        u_exact = np.exp(lam * t_hist[-1]) * np.sin(np.pi * (x - xmin) / (xmax - xmin))
        err = np.max(np.abs(u_hist[-1] - u_exact))
        errs.append(err)
        print(f"  N={N:3d}: L_inf = {err:.3e}")
    # Should drop to ~10^-14 or better
    assert errs[-1] < 1e-12


if __name__ == "__main__":
    print("Test: deterministic heat (single point)")
    test_deterministic_heat()
    print("\nTest: deterministic heat (N-refinement)")
    test_deterministic_convergence_N()
    print("\nDeterministic tests passed.")
