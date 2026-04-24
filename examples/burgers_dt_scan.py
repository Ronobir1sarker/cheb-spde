"""Find the real stability/accuracy boundary for both methods on Burgers.

Honest scan: vary dt, see where each method either blows up or drifts
from the fine-dt reference by more than 15% in integrated <u^2>.
"""
import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.linalg

from cheb_spde import ChebSPDESolver
from cheb_spde.chebyshev_grid import ChebyshevGrid
from cheb_spde.noise import clenshaw_curtis_weights


def burgers_nl_factory(solver, nu):
    D_full = solver.grid.D
    D_int = D_full[1:-1, 1:-1]; D_col_0 = D_full[1:-1, 0]; D_col_N = D_full[1:-1, -1]
    D2_full = solver.grid.D2
    D2_int = D2_full[1:-1, 1:-1]; D2_col_0 = D2_full[1:-1, 0]; D2_col_N = D2_full[1:-1, -1]
    def nl(u_int, t):
        gL = solver.gL(t); gR = solver.gR(t)
        u_x = D_int @ u_int + D_col_0 * gR + D_col_N * gL
        u_xx = D2_int @ u_int + D2_col_0 * gR + D2_col_N * gL
        return -u_int * u_x + (nu - 1.0) * u_xx
    return nl


class SIEMBurgers:
    def __init__(self, N, nu, sigma, seed):
        self.grid = ChebyshevGrid(N, -1.0, 1.0); self.N = N; self.nu = nu; self.sigma = sigma
        self.L_int = self.nu * self.grid.D2[1:-1, 1:-1]
        self.D_int = self.grid.D[1:-1, 1:-1]
        self.weights = clenshaw_curtis_weights(N, -1.0, 1.0)
        self.w_int = self.weights[1:-1]
        self.rng = np.random.default_rng(seed)
    def solve(self, T, dt):
        n_steps = int(round(T / dt)); m = self.N - 1
        A_lu = scipy.linalg.lu_factor(np.eye(m) - dt * self.L_int)
        v = 0.05 * np.sin(np.pi * self.grid.x[1:-1])
        tail_sq = np.zeros(self.N + 1); cnt = 0
        T_burn = min(4.0, 0.25 * T)
        for n in range(n_steps):
            u_x = self.D_int @ v
            z = self.rng.standard_normal(m)
            rhs = v + dt * (-v * u_x) + self.sigma * np.sqrt(dt / self.w_int) * z
            v = scipy.linalg.lu_solve(A_lu, rhs)
            if not np.all(np.isfinite(v)) or np.max(np.abs(v)) > 1e4:
                return None  # blown up
            t_now = (n + 1) * dt
            if t_now >= T_burn:
                full = np.zeros(self.N + 1); full[1:-1] = v
                tail_sq += full ** 2; cnt += 1
        return tail_sq / max(cnt, 1)


def exp_euler_burgers(N, nu, sigma, T, dt, seed):
    def gL(t): return 0.0 + 0.0 * t
    def gR(t): return 0.0 + 0.0 * t
    solver = ChebSPDESolver(
        N=N, xmin=-1.0, xmax=1.0,
        nonlinearity=None, gL=gL, gR=gR,
        noise_type="white", sigma=sigma,
        initial_condition=lambda x: 0.05 * np.sin(np.pi * x),
        seed=seed,
    )
    solver.nonlinearity = burgers_nl_factory(solver, nu)
    try:
        u_hist, t_hist = solver.solve(T=T, dt=dt, store_every=1)
    except Exception as e:
        return None
    if not np.all(np.isfinite(u_hist)):
        return None
    T_burn = min(4.0, 0.25 * T)
    mask = t_hist >= T_burn
    return np.mean(u_hist[mask] ** 2, axis=0)


def integrated_error(u_sq, ref, weights):
    """L2-weighted relative error:  ||f - f_ref||_w / ||f_ref||_w."""
    num = np.sqrt(np.sum(weights * (u_sq - ref) ** 2))
    den = np.sqrt(np.sum(weights * ref ** 2))
    return num / (den + 1e-30)


def main():
    N = 32; nu = 0.05; sigma = 0.5; T = 16.0
    n_runs = 12
    weights = clenshaw_curtis_weights(N, -1.0, 1.0)

    # Reference: SI-EM at small dt
    print(f"Reference run: SI-EM at dt = 5e-4, {n_runs} runs, T={T}")
    ref_acc = None
    for s in range(n_runs):
        em = SIEMBurgers(N, nu, sigma, seed=2000 + s)
        sq = em.solve(T=T, dt=5e-4)
        assert sq is not None
        if ref_acc is None: ref_acc = sq
        else: ref_acc = ref_acc + sq
    ref = ref_acc / n_runs

    # Scan both methods at increasing dt
    dts = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    exp_errs = []; exp_times = []; exp_blow = []
    em_errs = []; em_times = []; em_blow = []

    print(f"\n{'dt':>8} | {'ExpEuler err':>14} {'ExpEuler time':>14} | {'SI-EM err':>14} {'SI-EM time':>13}")
    print("-" * 80)
    for dt in dts:
        # ExpEuler
        t0 = time.perf_counter()
        acc = None; blew = False
        for s in range(n_runs):
            sq = exp_euler_burgers(N, nu, sigma, T, dt, seed=3000 + s)
            if sq is None:
                blew = True; break
            if acc is None: acc = sq
            else: acc = acc + sq
        exp_time = time.perf_counter() - t0
        if blew:
            exp_err = np.nan; exp_blow.append(True)
        else:
            exp_err = integrated_error(acc / n_runs, ref, weights)
            exp_blow.append(False)
        exp_errs.append(exp_err); exp_times.append(exp_time)

        # SI-EM
        t0 = time.perf_counter()
        acc = None; blew = False
        for s in range(n_runs):
            em = SIEMBurgers(N, nu, sigma, seed=3000 + s)
            sq = em.solve(T=T, dt=dt)
            if sq is None:
                blew = True; break
            if acc is None: acc = sq
            else: acc = acc + sq
        em_time = time.perf_counter() - t0
        if blew:
            em_err = np.nan; em_blow.append(True)
        else:
            em_err = integrated_error(acc / n_runs, ref, weights)
            em_blow.append(False)
        em_errs.append(em_err); em_times.append(em_time)

        exp_str = "BLOWUP" if exp_blow[-1] else f"{exp_err*100:8.2f}%"
        em_str  = "BLOWUP" if em_blow[-1]  else f"{em_err*100:8.2f}%"
        print(f"{dt:8.4f} | {exp_str:>14} {exp_time:14.3f} | {em_str:>14} {em_time:13.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    exp_arr = np.array(exp_errs); em_arr = np.array(em_errs)
    dts_arr = np.array(dts)
    ax.loglog(dts_arr, exp_arr, "o-", label="Cheb Exp-Euler (ours)", lw=2, ms=8)
    ax.loglog(dts_arr, em_arr, "s--", label="Cheb Semi-implicit EM", lw=2, ms=8)
    for dt, b in zip(dts, em_blow):
        if b:
            ax.axvline(dt, color="red", ls=":", alpha=0.3)
            ax.text(dt, 1e-2, "SI-EM\nblow-up", color="red", fontsize=8, ha="center")
    ax.axhline(0.15, color="gray", ls=":", label="15% threshold")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"weighted $L^2$ rel. err. in $\langle u^2\rangle$")
    ax.set_title(f"Stochastic Burgers: accuracy vs $\\Delta t$ (N={N}, nu={nu}, sigma={sigma})")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    out = Path(__file__).resolve().parent.parent / "figures" / "burgers_dt_scan.png"
    plt.tight_layout(); plt.savefig(out, dpi=140)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
