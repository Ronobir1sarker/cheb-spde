# cheb-spde

**A Chebyshev exponential-integrator solver for stochastic PDEs on bounded domains.**

Fast and memory-efficient Python package for simulating parabolic stochastic PDEs of the form

$$
\mathrm{d}u = \bigl( L u + N(u) \bigr)\, \mathrm{d}t + \mathrm{d}W_{\mathrm{noise}},
\qquad x \in [a, b], \quad u(a,t) = g_L(t),\ u(b,t) = g_R(t),
$$

on a bounded interval with time-dependent non-homogeneous Dirichlet boundary data and either additive space–time white noise or Ornstein–Uhlenbeck (temporally colored, spatially white) noise.

The scheme combines Chebyshev-collocation spatial discretization with a stochastic exponential-Euler time step. The linear dynamics are propagated exactly via matrix $\varphi$-functions evaluated by a Trefethen–Weideman–Schmelzer Talbot-contour quadrature; the stochastic convolution is sampled exactly in the operator eigenbasis; and a Clenshaw–Curtis quadrature rule gives the correct continuum noise scaling on the non-uniform grid.

## Key features

- **Spectral accuracy** on smooth fields through Chebyshev–Gauss–Lobatto collocation.
- **Exact linear propagation** via Talbot-contour matrix $\varphi_0, \varphi_1$ functions; no stability restriction from $\rho(L)$.
- **Exact stochastic-convolution sampling** in the eigenbasis of the interior Laplacian.
- **Correct continuum noise scaling** on the non-uniform grid through Clenshaw–Curtis weights.
- **Non-homogeneous, time-dependent Dirichlet boundary conditions** via boundary lifting with complex-step time derivatives.
- **Two noise models**: additive space–time white noise and Ornstein–Uhlenbeck (temporally colored, spatially white) noise.
- **~5× wall-clock speedup** over semi-implicit Euler–Maruyama on stochastic Allen–Cahn at matched statistical accuracy.
- **~8× stable-$\Delta t$ advantage** over semi-implicit Euler–Maruyama on stochastic Burgers.

## Installation

Requires Python 3.9+ with NumPy, SciPy, and Matplotlib.

```bash
git clone https://github.com/Ronobir1sarker/cheb-spde.git
cd cheb-spde
pip install -e .
```

Or install the test/dev dependencies as well:

```bash
pip install -e ".[test]"
```

## Quick start

Solve the stochastic Allen–Cahn equation $\mathrm{d}u = (u_{xx} + u - u^3)\,\mathrm{d}t + \sigma\,\mathrm{d}W$ on $[-1, 1]$ with zero Dirichlet boundary data and random initial condition:

```python
import numpy as np
from cheb_spde import ChebSPDESolver

def u0(x):
    return 0.1 * np.random.randn(len(x))

def nonlinearity(u):
    return u - u**3

solver = ChebSPDESolver(
    N=32, xmin=-1.0, xmax=1.0,
    nonlinearity=nonlinearity,
    gL=lambda t: 0.0,
    gR=lambda t: 0.0,
    sigma=0.3,
    dt=0.02,
    seed=42,
)

u_final = solver.solve(u0(solver.grid.x), T=10.0)
```

## Examples

The `examples/` directory contains four end-to-end runnable scripts that reproduce every figure in the accompanying paper:

| Script | What it produces |
|---|---|
| `benchmark_linear.py` | Stationary-variance validation + wall-clock benchmark vs SI-EM |
| `allen_cahn_demo.py` | Stochastic Allen–Cahn trajectory + ensemble statistics |
| `burgers_demo.py` | Stochastic Burgers trajectory and snapshots |
| `burgers_dt_scan.py` | Stable-$\Delta t$ scan vs semi-implicit Euler–Maruyama |

Run any example directly:

```bash
python examples/allen_cahn_demo.py
```

Each script writes a PNG to `figures/`.

## Validation

Run the full test suite to verify installation and correctness:

```bash
pytest tests/ -v
```

Three tests cover:

1. **`test_deterministic.py`** — spectral convergence on the heat equation ($L^\infty$ error drops to $3 \times 10^{-15}$ at $N = 16$).
2. **`test_stationary_variance.py`** — linear SPDE stationary variance matches the exact Green's function $\sigma^2(1-x^2)/4$ to 0.6% Monte-Carlo error.
3. **`test_ou_noise.py`** — OU-noise autocorrelation matches $e^{-|s|/\tau}$ to 0.036 absolute error and OU-driven SPDE converges under $\Delta t$ refinement.

All tests complete in under 2 minutes on a laptop.

## Package structure

```
cheb-spde/
├── cheb_spde/              # Main package
│   ├── __init__.py
│   ├── chebyshev_grid.py   # CGL nodes + differentiation matrices
│   ├── phi_functions.py    # Talbot-contour matrix phi-functions
│   ├── noise.py            # Clenshaw–Curtis weights, white/OU noise
│   └── spde_solver.py      # Main ChebSPDESolver class
├── tests/                  # pytest suite (3 validation tests)
├── examples/               # 4 demonstration scripts
├── figures/                # Pre-rendered figures from the paper
├── docs/                   # Additional documentation
├── pyproject.toml          # Package metadata and dependencies
├── LICENSE.txt             # MIT License
└── README.md               # This file
```

## Method summary

The scheme advances the interior unknown $v_n \in \mathbb{R}^{N-1}$ (after boundary lifting $u = v + \ell$) as

$$
v_{n+1} = \varphi_0(h L_{\mathrm{int}})\, v_n + h\, \varphi_1(h L_{\mathrm{int}})\, \tilde F(v_n, t_n) + \xi_n,
$$

where

- $\varphi_0(h L_{\mathrm{int}}) = \exp(h L_{\mathrm{int}})$ and $\varphi_1(z) = (e^z - 1)/z$ are computed as matrix functions via a Trefethen–Weideman–Schmelzer Talbot contour (32–64 quadrature nodes; one dense LU solve per node; performed once at setup).
- $\tilde F$ collects the lifted nonlinearity, boundary columns of $L$ applied to prescribed boundary values, and $\partial \ell/\partial t$ (evaluated by complex-step differentiation at machine precision).
- $\xi_n \sim \mathcal{N}(0, C)$ is a Gaussian random vector sampled exactly by Cholesky factorization of the covariance $C = \sigma^2 \int_0^h e^{s L_{\mathrm{int}}} W^{-1} e^{s L_{\mathrm{int}}^T}\,\mathrm{d}s$ in the eigenbasis of $L_{\mathrm{int}}$; $W = \mathrm{diag}(w_1, \ldots, w_{N-1})$ is the diagonal of interior Clenshaw–Curtis weights.

Per-step cost: two $\mathcal{O}(N^2)$ matrix-vector products and one Cholesky-factor application. Setup cost: $\mathcal{O}(N^3)$ once per $(L_{\mathrm{int}}, \Delta t)$ pair.

## References

The method combines ingredients from:

- Cox & Matthews, *Exponential time differencing for stiff systems*, J. Comput. Phys. 176 (2002) 430–455.
- Kassam & Trefethen, *Fourth-order time stepping for stiff PDEs*, SIAM J. Sci. Comput. 26 (2005) 1214–1233.
- Trefethen, Weideman & Schmelzer, *Talbot quadratures and rational approximations*, BIT 46 (2006) 653–670.
- Lord & Tambue, *Stochastic exponential integrators for the finite element discretization of SPDEs with additive noise*, IMA J. Numer. Anal. 33 (2013) 515–543.
- Waldvogel, *Fast construction of the Fejér and Clenshaw–Curtis quadrature rules*, BIT 46 (2006) 195–202.

## Citation

If you use this code in a publication, please cite the accompanying SoftwareX paper:

```bibtex
@article{chebspde2026,
  author  = {Ronobir Chandra Sarker},
  title   = {cheb-spde: A Chebyshev exponential-integrator solver for stochastic PDEs on bounded domains},
  journal = {SoftwareX},
  year    = {2026},
  note    = {In preparation.},
}
```

## License

MIT License — see `LICENSE.txt`.

## Contact

Issues and pull requests welcome on GitHub. For other questions, contact ronobir.sarker@gmail.com.
