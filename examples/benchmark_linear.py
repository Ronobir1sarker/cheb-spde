"""Benchmark: stability / largest-allowed-dt vs semi-implicit Euler-Maruyama.

Problem: linear SPDE on [-1, 1] with Dirichlet BCs,
    du = u_xx dt + sigma dW.

We compare two integrators on the Chebyshev grid:

1. Our exponential-Euler (ChebSPDESolver): exact in the linear part.
2. Semi-implicit Euler-Maruyama:
      v_{n+1} = (I - dt L_int)^{-1} (v_n + sqrt(dt) W^{-1/2} z_n * sigma).

For each integrator we find the largest dt at which the stationary variance
stays within 5% of the exact curve, and measure wall-clock for the same T.
"""
import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cheb_spde import ChebSPDESolver
from cheb_spde.chebyshev_grid import ChebyshevGrid
from cheb_spde.noise import clenshaw_curtis_weights


class SemiImplicitEMCheb:
    """Semi-implicit Euler-Maruyama baseline on Chebyshev grid.

        v_{n+1} = (I - dt L_int)^{-1} (v_n + sigma sqrt(dt/w) z_n),

    implemented with a cached LU factorisation of (I - dt L_int).
    """
    def __init__(self, N, xmin, xmax, sigma, seed=0):
        self.grid = ChebyshevGrid(N, xmin, xmax)
        self.N = N
        self.L_int = self.grid.D2[1:-1, 1:-1].copy()
        self.weights = clenshaw_curtis_weights(N, xmin, xmax)
        self.w_int = self.weights[1:-1]
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self._A_lu = None
        self._dt = None

    def prepare(self, dt):
        if self._dt == dt:
            return
        self._dt = dt
        import scipy.linalg
        m = self.L_int.shape[0]
        self._A_lu = scipy.linalg.lu_factor(np.eye(m) - dt * self.L_int)

    def solve(self, T, dt, store_every=1):
        import scipy.linalg
        self.prepare(dt)
        n_steps = int(round(T / dt))
        n_store = n_steps // store_every + 1
        u = np.zeros((n_store, self.N + 1))
        t_hist = np.zeros(n_store)
        v = np.zeros(self.N - 1)
        store_idx = 1
        for n in range(n_steps):
            z = self.rng.standard_normal(self.N - 1)
            rhs = v + self.sigma * np.sqrt(dt / self.w_int) * z
            v = scipy.linalg.lu_solve(self._A_lu, rhs)
            if (n + 1) % store_every == 0 and store_idx < n_store:
                full = np.zeros(self.N + 1); full[1:-1] = v
                u[store_idx] = full
                t_hist[store_idx] = (n + 1) * dt
                store_idx += 1
        return u[:store_idx], t_hist[:store_idx]


def stationary_variance_error(integrator_name, solver, T, dt, T_burn, n_runs, xmin=-1.0, xmax=1.0):
    """Return (max relative error in u^2 at interior, wall-clock per run)."""
    sigma = solver.sigma if hasattr(solver, "sigma") else None
    times = []
    u_sq_total = None
    x = None
    for run in range(n_runs):
        # Re-seed by reconstructing rng
        solver.rng = np.random.default_rng(run + 1000)
        t0 = time.perf_counter()
        u_hist, t_hist = solver.solve(T=T, dt=dt, store_every=1)
        times.append(time.perf_counter() - t0)
        mask = t_hist >= T_burn
        u_sq = np.mean(u_hist[mask] ** 2, axis=0)
        if u_sq_total is None:
            u_sq_total = u_sq
        else:
            u_sq_total += u_sq
        x = solver.grid.x
    u_sq_avg = u_sq_total / n_runs
    u_sq_exact = sigma ** 2 * (1 - x ** 2) / 4.0
    # Relative error at the middle third of the interval (avoid tiny-variance boundary)
    mid = (x > xmin * 0.5) & (x < xmax * 0.5)
    rel = np.max(np.abs(u_sq_avg[mid] - u_sq_exact[mid]) / u_sq_exact[mid])
    return rel, float(np.mean(times)), x, u_sq_avg, u_sq_exact


def run_benchmark():
    xmin, xmax = -1.0, 1.0
    N = 24
    sigma = 0.5
    T = 30.0
    T_burn = 5.0
    n_runs = 20

    dts = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]

    exp_results = []
    em_results = []

    print(f"\nN = {N}, sigma = {sigma}, T = {T}, {n_runs} runs per point\n")
    print(f"{'dt':>8} | {'ExpEuler err':>14} {'ExpEuler time(s)':>18} | {'SemiIMEX err':>14} {'SemiIMEX time(s)':>18}")
    print("-" * 90)

    def u0(x): return np.zeros_like(x)
    def nl(u_int, t): return np.zeros_like(u_int)
    gL = lambda t: 0.0 + 0.0 * t
    gR = lambda t: 0.0 + 0.0 * t

    for dt in dts:
        # Exponential Euler
        exp_solver = ChebSPDESolver(
            N=N, xmin=xmin, xmax=xmax,
            nonlinearity=nl, gL=gL, gR=gR,
            noise_type="white", sigma=sigma,
            initial_condition=u0, seed=0,
        )
        exp_rel, exp_time, *_ = stationary_variance_error(
            "ExpEuler", exp_solver, T, dt, T_burn, n_runs)
        exp_results.append((dt, exp_rel, exp_time))

        # Semi-implicit EM
        em_solver = SemiImplicitEMCheb(N, xmin, xmax, sigma, seed=0)
        em_rel, em_time, *_ = stationary_variance_error(
            "SemiIMEX", em_solver, T, dt, T_burn, n_runs)
        em_results.append((dt, em_rel, em_time))

        print(f"{dt:8.4f} | {exp_rel*100:13.2f}% {exp_time:18.3f} | {em_rel*100:13.2f}% {em_time:18.3f}")

    # Plot
    exp_dt = np.array([r[0] for r in exp_results])
    exp_err = np.array([r[1] for r in exp_results])
    exp_time = np.array([r[2] for r in exp_results])
    em_dt = np.array([r[0] for r in em_results])
    em_err = np.array([r[1] for r in em_results])
    em_time = np.array([r[2] for r in em_results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].loglog(exp_dt, exp_err, "o-", label="Cheb Exp-Euler (ours)", lw=2, ms=7)
    axes[0].loglog(em_dt, em_err, "s--", label="Cheb Semi-implicit EM", lw=2, ms=7)
    axes[0].axhline(0.05, color="gray", ls=":", label="5% threshold")
    axes[0].set_xlabel(r"$\Delta t$")
    axes[0].set_ylabel("max rel. error in stationary variance")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[0].set_title("(a) Accuracy vs time step")

    axes[1].loglog(exp_time, exp_err, "o-", label="Cheb Exp-Euler (ours)", lw=2, ms=7)
    axes[1].loglog(em_time, em_err, "s--", label="Cheb Semi-implicit EM", lw=2, ms=7)
    axes[1].axhline(0.05, color="gray", ls=":", label="5% threshold")
    axes[1].set_xlabel("wall-clock per run (s)")
    axes[1].set_ylabel("max rel. error in stationary variance")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    axes[1].set_title("(b) Cost vs accuracy")

    plt.tight_layout()
    out = Path(__file__).resolve().parent.parent / "figures" / "benchmark_linear.png"
    plt.savefig(out, dpi=140)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run_benchmark()
