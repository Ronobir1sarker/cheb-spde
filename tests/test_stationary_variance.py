"""Test 2: Stationary variance of the linear SPDE

    du = u_xx dt + sigma dW,   u(-1,t) = u(1,t) = 0.

Exact stationary variance:

    <u(x)^2>_inf = (sigma^2 / 2) * G(x, x)
                 = (sigma^2 / 2) * (1 - x^2) / 2
                 = sigma^2 * (1 - x^2) / 4,

where G(x,x) = (1 - x^2)/2 is the diagonal of the Dirichlet Green's
function of -d^2/dx^2 on [-1, 1].

If the Clenshaw-Curtis noise scaling AND the stochastic-convolution
Cholesky are both correct, an ensemble of long runs of our solver
should reproduce this curve to within O(1/sqrt(M)) Monte Carlo error.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cheb_spde import ChebSPDESolver


def run_one(seed, N, sigma, T, dt, T_burn, xmin=-1.0, xmax=1.0):
    """Run solver once, return time-averaged u^2 after burn-in."""
    def u0(x): return np.zeros_like(x)
    def nl(u_int, t): return np.zeros_like(u_int)
    gL = lambda t: 0.0 + 0.0 * t
    gR = lambda t: 0.0 + 0.0 * t

    solver = ChebSPDESolver(
        N=N, xmin=xmin, xmax=xmax,
        nonlinearity=nl, gL=gL, gR=gR,
        noise_type="white", sigma=sigma,
        initial_condition=u0, seed=seed,
    )
    u_hist, t_hist = solver.solve(T=T, dt=dt, store_every=1)
    # Time-average u^2 over the stationary portion
    mask = t_hist >= T_burn
    u_sq_mean = np.mean(u_hist[mask] ** 2, axis=0)
    return solver.grid.x, u_sq_mean


def ensemble_variance(N, sigma, T, dt, T_burn, n_runs):
    x = None
    u_sq = None
    for seed in range(n_runs):
        x_, u_sq_ = run_one(seed, N, sigma, T, dt, T_burn)
        if u_sq is None:
            u_sq = u_sq_
        else:
            u_sq = u_sq + u_sq_
        x = x_
    return x, u_sq / n_runs


def test_stationary_variance():
    N = 24
    sigma = 0.5
    T = 40.0
    dt = 0.02
    T_burn = 5.0
    n_runs = 40

    print(f"  Running {n_runs} realizations, T={T}, dt={dt}, burn-in={T_burn}, N={N}, sigma={sigma}...")
    x, u_sq_emp = ensemble_variance(N, sigma, T, dt, T_burn, n_runs)

    # Exact stationary variance on [-1, 1]
    u_sq_exact = sigma ** 2 * (1.0 - x ** 2) / 4.0

    # Compare at interior nodes (BCs force exact zero at boundaries)
    interior = slice(1, -1)
    err_abs = np.max(np.abs(u_sq_emp[interior] - u_sq_exact[interior]))
    # Relative error at the peak (x = 0)
    peak = sigma ** 2 / 4.0
    idx_mid = np.argmin(np.abs(x))
    rel_peak = abs(u_sq_emp[idx_mid] - u_sq_exact[idx_mid]) / peak

    print(f"  Peak variance: empirical = {u_sq_emp[idx_mid]:.4f}, exact = {peak:.4f}")
    print(f"  Relative error at peak: {rel_peak*100:.2f}%")
    print(f"  Max absolute error (interior): {err_abs:.4f}")

    # Save diagnostic plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, u_sq_exact, 'k-', lw=2, label='exact: $\\sigma^2(1-x^2)/4$')
    ax.plot(x, u_sq_emp, 'o', mfc='none', ms=6, label=f'empirical ({n_runs} runs, T={T})')
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\langle u(x)^2\rangle_\infty$')
    ax.set_title(f'Stationary variance: linear SPDE on [-1,1], $\\sigma={sigma}$')
    ax.grid(alpha=0.3)
    ax.legend()
    out = Path(__file__).resolve().parent.parent / "figures" / "test_stationary_variance.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"  Saved: {out}")

    # With 40 runs and a variance that's time-integrated, Monte Carlo error should be
    # well under 10% relative. Tighten this threshold as a sanity bound.
    assert rel_peak < 0.15, f"Stationary variance off by {rel_peak*100:.1f}%; noise scaling likely wrong"


if __name__ == "__main__":
    print("Test: stationary variance vs Green's function")
    test_stationary_variance()
    print("\nPassed.")
