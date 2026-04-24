# Changelog

All notable changes to cheb-spde are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [semantic versioning](https://semver.org/).

## [1.0.0] - 2026

Initial public release accompanying the SoftwareX submission.

### Added
- Chebyshev-Gauss-Lobatto grid and differentiation matrices (`chebyshev_grid.py`).
- Matrix phi-functions evaluated by Trefethen-Weideman-Schmelzer Talbot contour quadrature (`phi_functions.py`).
- Clenshaw-Curtis quadrature weights and white/OU noise samplers (`noise.py`).
- Main `ChebSPDESolver` class implementing stochastic exponential-Euler with exact stochastic-convolution sampling (`spde_solver.py`).
- Non-homogeneous, time-dependent Dirichlet boundary lifting with complex-step time derivative.
- Validation test suite: deterministic spectral convergence, linear-SPDE stationary variance vs Green's function, OU noise autocorrelation and stationary variance.
- Example scripts: linear benchmark vs semi-implicit Euler-Maruyama, stochastic Allen-Cahn, stochastic Burgers, Burgers dt-scan.
- Full installation, usage, and API documentation in README.
