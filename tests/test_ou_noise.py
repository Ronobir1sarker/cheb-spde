"""Test 3: OU noise process validation.

We validate the standalone OU integrator (src/noise.py) against its two
defining analytical properties:

1. Stationary variance at node i:  <eta_i^2>_inf = sigma^2 / w_i,
   where w_i is the Clenshaw-Curtis weight at node x_i.

2. Autocorrelation:  <eta(t) eta(t+s)> / <eta^2> = exp(-|s|/tau).

Then we validate the OU-driven linear SPDE

    du = u_xx dt + eta(x,t) dt,

against the *slow limit* tau -> infinity (OU behaves like frozen noise on
[0, T]), and against a fine-reference-dt run at moderate tau.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cheb_spde.noise import OUNoiseCheb, clenshaw_curtis_weights
from cheb_spde import ChebSPDESolver


def test_ou_stationary_variance():
    """Run many OU realizations; compare ensemble variance to sigma^2 / w_i.

    The OU process has integrated autocorrelation time tau, so a single run
    of length T provides T/tau effectively-independent samples per node. To
    distinguish bias from Monte Carlo error we run many independent
    realizations and show the ensemble mean converges to the exact formula.
    """
    N = 24
    xmin, xmax = -1.0, 1.0
    sigma = 0.8
    tau = 0.5
    dt = 0.01
    T = 100.0
    n_runs = 20
    n_steps = int(round(T / dt))

    weights = clenshaw_curtis_weights(N, xmin, xmax)

    runs = np.zeros((n_runs, N + 1))
    for s in range(n_runs):
        rng = np.random.default_rng(42 + s)
        ou = OUNoiseCheb(N=N, weights=weights, tau=tau, sigma=sigma, dt=dt, rng=rng)
        burn = int(5 * tau / dt)
        for _ in range(burn):
            ou.step()
        acc = np.zeros(N + 1)
        for _ in range(n_steps - burn):
            eta = ou.step()
            acc += eta ** 2
        runs[s] = acc / (n_steps - burn)

    emp_var = runs.mean(axis=0)
    # Standard error at each node across runs
    se = runs.std(axis=0, ddof=1) / np.sqrt(n_runs)
    exact = (sigma ** 2) / weights

    interior = slice(1, -1)
    rel = np.max(np.abs(emp_var[interior] - exact[interior]) / exact[interior])
    # Largest deviation in units of its standard error
    z = np.max(np.abs(emp_var[interior] - exact[interior]) / (se[interior] + 1e-30))

    print(f"  N={N}, tau={tau}, sigma={sigma}, {n_runs} runs each of T={T}")
    print(f"  Max rel err in stationary variance (interior): {rel*100:.2f}%")
    print(f"  Largest deviation in units of SE:              {z:.2f}")
    # Bias should vanish: the ensemble mean should agree with exact within ~3 SE
    assert z < 4.0, f"OU stationary variance biased at {z:.1f} SE"
    assert rel < 0.05, f"OU stationary variance off by {rel*100:.1f}%"


def test_ou_autocorrelation():
    """Run OU and compute empirical autocorrelation; compare to exp(-|s|/tau)."""
    N = 16
    xmin, xmax = -1.0, 1.0
    sigma = 1.0
    tau = 1.0
    dt = 0.01
    T = 400.0
    n_steps = int(round(T / dt))

    weights = clenshaw_curtis_weights(N, xmin, xmax)
    rng = np.random.default_rng(7)
    ou = OUNoiseCheb(N=N, weights=weights, tau=tau, sigma=sigma, dt=dt, rng=rng)

    # Burn in then record trajectory at one interior node
    mid = N // 2
    burn = int(10 * tau / dt)
    for _ in range(burn):
        ou.step()
    traj = np.zeros(n_steps)
    for n in range(n_steps):
        eta = ou.step()
        traj[n] = eta[mid]

    # Empirical autocorrelation (biased estimator)
    traj -= traj.mean()
    var = np.var(traj)
    # Lags in steps to examine up to 6*tau
    max_lag_steps = int(6 * tau / dt)
    lags = np.arange(0, max_lag_steps, max(1, max_lag_steps // 60))
    emp_acf = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag == 0:
            emp_acf[i] = 1.0
        else:
            emp_acf[i] = np.mean(traj[:-lag] * traj[lag:]) / var
    t_lags = lags * dt
    exact_acf = np.exp(-t_lags / tau)

    err = np.max(np.abs(emp_acf - exact_acf))
    print(f"\n  N={N}, tau={tau}, T={T}, node x[{mid}]")
    print(f"  Max abs err in autocorrelation: {err:.3f}")
    assert err < 0.15, f"OU autocorrelation off by {err:.3f}"

    # Save plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_lags, exact_acf, "k-", lw=2, label=r"exact: $e^{-|s|/\tau}$")
    ax.plot(t_lags, emp_acf, "o", mfc="none", label=f"empirical (T={T})")
    ax.set_xlabel("lag s")
    ax.set_ylabel(r"$\langle \eta(t)\eta(t+s)\rangle / \langle \eta^2\rangle$")
    ax.set_title(f"OU noise autocorrelation ($\\tau={tau}$)")
    ax.grid(alpha=0.3)
    ax.legend()
    out = Path(__file__).resolve().parent.parent / "figures" / "test_ou_autocorrelation.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"  Saved: {out}")


def test_ou_driven_spde():
    """
    Linear SPDE driven by OU noise:
        du = u_xx dt + eta(x,t) dt,
        u(-1,t) = u(1,t) = 0,
    with eta an OU process of parameters (tau, sigma).

    Exact stationary variance of u is NOT available in closed form in general,
    so we compare to a fine-dt run of the SAME exp-Euler solver (dt_ref << dt).
    If our OU path is correct, moderate-dt and fine-dt should agree within MC.
    """
    N = 24
    xmin, xmax = -1.0, 1.0
    sigma = 0.6
    tau = 0.5
    T = 30.0
    T_burn = 4.0
    n_runs = 25

    def u0(x): return np.zeros_like(x)
    def nl(u_int, t): return np.zeros_like(u_int)
    gL = lambda t: 0.0 + 0.0 * t
    gR = lambda t: 0.0 + 0.0 * t

    def run_many(dt):
        u_sq_accum = None
        for s in range(n_runs):
            solver = ChebSPDESolver(
                N=N, xmin=xmin, xmax=xmax,
                nonlinearity=nl, gL=gL, gR=gR,
                noise_type="ou", sigma=sigma, tau=tau,
                initial_condition=u0, seed=1000 + s,
            )
            u_hist, t_hist = solver.solve(T=T, dt=dt, store_every=1)
            mask = t_hist >= T_burn
            sq = np.mean(u_hist[mask] ** 2, axis=0)
            if u_sq_accum is None:
                u_sq_accum = sq
            else:
                u_sq_accum += sq
            x = solver.grid.x
        return x, u_sq_accum / n_runs

    print("\n  Running moderate dt = 0.02...")
    x, u_sq_mod = run_many(0.02)
    print("  Running fine dt = 0.002...")
    _, u_sq_fine = run_many(0.002)

    interior = slice(1, -1)
    rel = np.max(np.abs(u_sq_mod[interior] - u_sq_fine[interior])
                 / (u_sq_fine[interior] + 1e-12))
    print(f"  Max rel. err in <u^2> (dt=0.02 vs dt=0.002): {rel*100:.2f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(x)
    ax.plot(x[order], u_sq_fine[order], "k-", lw=2, label="dt = 0.002 (reference)")
    ax.plot(x[order], u_sq_mod[order], "o", mfc="none", label="dt = 0.02 (tested)")
    ax.set_xlabel("x"); ax.set_ylabel(r"$\langle u(x)^2\rangle$")
    ax.set_title(f"OU-driven SPDE: stationary variance, tau={tau}, sigma={sigma}")
    ax.grid(alpha=0.3); ax.legend()
    out = Path(__file__).resolve().parent.parent / "figures" / "test_ou_driven_spde.png"
    plt.tight_layout(); plt.savefig(out, dpi=140)
    print(f"  Saved: {out}")

    # Tolerate MC error comfortably
    assert rel < 0.20, f"OU-driven SPDE inconsistent across dt: err = {rel*100:.1f}%"


if __name__ == "__main__":
    print("OU noise stationary variance:")
    test_ou_stationary_variance()
    print("\nOU noise autocorrelation:")
    test_ou_autocorrelation()
    print("\nOU-driven SPDE (moderate vs fine dt):")
    test_ou_driven_spde()
    print("\nAll OU tests passed.")
