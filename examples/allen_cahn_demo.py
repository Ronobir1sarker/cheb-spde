"""Headline test: stochastic Allen-Cahn on [-1, 1] with Dirichlet BCs.

    du = (u_xx + u - u^3) dt + sigma dW,   u(-1,t) = u(1,t) = 0.

Deterministic Allen-Cahn on this interval with Dirichlet BCs has three
equilibria: u = 0 (unstable) and two non-trivial localized minimizers
u_+(x), u_-(x) = -u_+(x) (stable). With additive noise, the system
hops rarely between +u_+ and -u_+ (a Kramers-type escape); on moderate
time scales the solution settles near one of the two stable branches.

We:
  1. Show a single long trajectory transitioning out of u = 0 and
     relaxing toward one of the stable branches (visibility only).
  2. Compare the time-averaged <u(x)> and <u(x)^2> against a reference
     run done with semi-implicit Euler-Maruyama at dt = 5e-4.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cheb_spde import ChebSPDESolver


def allen_cahn_nl(u_int, t):
    return u_int - u_int ** 3


def run_ac_solver(N, sigma, T, dt, seed=0, store_every=5, u0_fn=None):
    def gL(t): return 0.0 + 0.0 * t
    def gR(t): return 0.0 + 0.0 * t
    solver = ChebSPDESolver(
        N=N, xmin=-1.0, xmax=1.0,
        nonlinearity=allen_cahn_nl, gL=gL, gR=gR,
        noise_type="white", sigma=sigma,
        initial_condition=u0_fn if u0_fn is not None else (lambda x: 0.01 * np.sin(np.pi * (x + 1) / 2)),
        seed=seed,
    )
    return solver, *solver.solve(T=T, dt=dt, store_every=store_every)


def plot_spacetime():
    """Single trajectory: transition from u=0 toward a stable branch."""
    N = 32
    sigma = 0.3
    T = 60.0
    dt = 0.02

    solver, u_hist, t_hist = run_ac_solver(N, sigma, T, dt, seed=3, store_every=2)
    x = solver.grid.x

    # Sort x for plotting (CGL order is reversed)
    order = np.argsort(x)
    x_sorted = x[order]
    u_sorted = u_hist[:, order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].pcolormesh(x_sorted, t_hist, u_sorted, shading='auto', cmap='RdBu_r',
                            vmin=-1.1, vmax=1.1)
    fig.colorbar(im, ax=axes[0], label='u(x,t)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title(f'(a) Space-time plot: stochastic Allen-Cahn, $\\sigma={sigma}$')

    # Plot snapshots
    snap_times = [0.0, 2.0, 10.0, 30.0, 60.0]
    for st in snap_times:
        idx = np.argmin(np.abs(t_hist - st))
        axes[1].plot(x_sorted, u_sorted[idx], label=f't = {t_hist[idx]:.1f}')
    axes[1].axhline(0, color='gray', lw=0.5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u(x)')
    axes[1].set_title('(b) Snapshots: transition out of $u \\equiv 0$')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent.parent / "figures" / "allen_cahn_trajectory.png"
    plt.savefig(out, dpi=140)
    print(f"Saved: {out}")


def compare_with_reference():
    """
    Ensemble statistics at moderate T, comparing our exp-Euler at dt=0.02
    against a fine-dt semi-implicit EM reference at dt=0.001.
    """
    import time
    from cheb_spde.chebyshev_grid import ChebyshevGrid
    from cheb_spde.noise import clenshaw_curtis_weights
    import scipy.linalg

    N = 32
    sigma = 0.3
    T = 20.0
    T_burn = 2.0
    n_runs = 24

    # -- Our exp-Euler at moderate dt --
    dt_exp = 0.02
    u_sq_exp = None
    u_mean_exp = None
    t_exp_total = 0.0
    for s in range(n_runs):
        def gL(t): return 0.0 + 0.0 * t
        def gR(t): return 0.0 + 0.0 * t
        solver = ChebSPDESolver(
            N=N, xmin=-1.0, xmax=1.0,
            nonlinearity=allen_cahn_nl, gL=gL, gR=gR,
            noise_type="white", sigma=sigma,
            initial_condition=lambda x: 0.01 * np.sin(np.pi * (x + 1) / 2),
            seed=s * 7 + 1,
        )
        t0 = time.perf_counter()
        u_hist, t_hist = solver.solve(T=T, dt=dt_exp, store_every=1)
        t_exp_total += time.perf_counter() - t0
        mask = t_hist >= T_burn
        # Use |u|^2 and |u| (sign-flipped for symmetric wells, fold via sign of integral)
        sgn = np.sign(np.trapezoid(u_hist[mask], solver.grid.x, axis=1) + 1e-30)
        u_folded = u_hist[mask] * sgn[:, None]
        sq = np.mean(u_folded ** 2, axis=0)
        m = np.mean(u_folded, axis=0)
        if u_sq_exp is None:
            u_sq_exp = sq
            u_mean_exp = m
        else:
            u_sq_exp += sq
            u_mean_exp += m
        x_grid = solver.grid.x
    u_sq_exp /= n_runs
    u_mean_exp /= n_runs

    # -- Reference: semi-implicit EM at fine dt --
    class SI_EM_AC:
        def __init__(self, N, sigma, seed):
            self.grid = ChebyshevGrid(N, -1.0, 1.0)
            self.N = N
            self.L_int = self.grid.D2[1:-1, 1:-1]
            self.weights = clenshaw_curtis_weights(N, -1.0, 1.0)
            self.w_int = self.weights[1:-1]
            self.sigma = sigma
            self.rng = np.random.default_rng(seed)
        def solve(self, T, dt, store_every):
            n_steps = int(round(T / dt))
            m = self.N - 1
            A_lu = scipy.linalg.lu_factor(np.eye(m) - dt * self.L_int)
            v = 0.01 * np.sin(np.pi * (self.grid.x[1:-1] + 1) / 2)
            n_store = n_steps // store_every + 1
            u = np.zeros((n_store, self.N + 1))
            t_hist = np.zeros(n_store)
            idx = 1
            for n in range(n_steps):
                nl = v - v ** 3
                z = self.rng.standard_normal(m)
                rhs = v + dt * nl + self.sigma * np.sqrt(dt / self.w_int) * z
                v = scipy.linalg.lu_solve(A_lu, rhs)
                if (n + 1) % store_every == 0 and idx < n_store:
                    full = np.zeros(self.N + 1); full[1:-1] = v
                    u[idx] = full
                    t_hist[idx] = (n + 1) * dt
                    idx += 1
            return u[:idx], t_hist[:idx]

    dt_ref = 0.001
    store_every_ref = 20  # same storage cadence as exp solver
    u_sq_ref = None
    u_mean_ref = None
    t_ref_total = 0.0
    for s in range(n_runs):
        em = SI_EM_AC(N, sigma, seed=s * 7 + 1)
        t0 = time.perf_counter()
        u_hist, t_hist = em.solve(T=T, dt=dt_ref, store_every=store_every_ref)
        t_ref_total += time.perf_counter() - t0
        mask = t_hist >= T_burn
        sgn = np.sign(np.trapezoid(u_hist[mask], em.grid.x, axis=1) + 1e-30)
        u_folded = u_hist[mask] * sgn[:, None]
        sq = np.mean(u_folded ** 2, axis=0)
        m = np.mean(u_folded, axis=0)
        if u_sq_ref is None:
            u_sq_ref = sq
            u_mean_ref = m
        else:
            u_sq_ref += sq
            u_mean_ref += m
        x_grid = em.grid.x
    u_sq_ref /= n_runs
    u_mean_ref /= n_runs

    # Diff
    rel_err_mean = np.max(np.abs(u_mean_exp - u_mean_ref)) / np.max(np.abs(u_mean_ref) + 1e-12)
    rel_err_sq = np.max(np.abs(u_sq_exp - u_sq_ref)) / np.max(np.abs(u_sq_ref) + 1e-12)

    print(f"\nAllen-Cahn ensemble comparison ({n_runs} runs each, T={T})")
    print(f"  Our method:   dt = {dt_exp},  wall clock = {t_exp_total:.2f}s total")
    print(f"  Reference:    dt = {dt_ref}, wall clock = {t_ref_total:.2f}s total")
    print(f"  Speedup:      {t_ref_total / t_exp_total:.1f}x")
    print(f"  Max rel. err in <u(x)>:    {rel_err_mean*100:.2f}%")
    print(f"  Max rel. err in <u(x)^2>:  {rel_err_sq*100:.2f}%")

    order = np.argsort(x_grid)
    x_s = x_grid[order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(x_s, u_mean_ref[order], 'k-', lw=2, label=f'SI-EM ref (dt={dt_ref})')
    axes[0].plot(x_s, u_mean_exp[order], 'o', mfc='none', label=f'Cheb Exp-Euler (dt={dt_exp})')
    axes[0].set_xlabel('x'); axes[0].set_ylabel(r'$\langle |u|(x)\rangle$')
    axes[0].set_title('(a) Sign-folded mean (stable-branch profile)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(x_s, u_sq_ref[order], 'k-', lw=2, label=f'SI-EM ref (dt={dt_ref})')
    axes[1].plot(x_s, u_sq_exp[order], 'o', mfc='none', label=f'Cheb Exp-Euler (dt={dt_exp})')
    axes[1].set_xlabel('x'); axes[1].set_ylabel(r'$\langle u(x)^2\rangle$')
    axes[1].set_title('(b) Second moment')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent.parent / "figures" / "allen_cahn_compare.png"
    plt.savefig(out, dpi=140)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Allen-Cahn trajectory plot...")
    plot_spacetime()
    print("\nAllen-Cahn statistics vs reference...")
    compare_with_reference()
