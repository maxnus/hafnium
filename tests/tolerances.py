"""Named tolerances for tests (CONVENTIONS.md §8). Never use inline literals."""

ENERGY_TOL = 1e-10
INTEGRAL_TOL = 1e-12
GRID_TOL = 1e-6
# Kernel results versus NumPy: the two sum in different orders (sequential
# loop versus BLAS), so they agree only to about n * machine epsilon.
KERNEL_REL_TOL = 1e-12
# Block-structure round trips and algebra identities on small matrices.
STRUCTURE_TOL = 1e-10
# Eigen-decompositions: residuals of F C - S C e and orthonormality.
EIGEN_TOL = 1e-9
# The Boys function against arbitrary-precision quadrature (ADR-0008).
BOYS_REL_TOL = 1e-14
# Four-decimal literature tables (Szabo and Ostlund).
LITERATURE_TOL = 5e-5
# Spectra of rotated and translated copies of a molecule.
SPECTRUM_TOL = 1e-10
# Fitted four-centre integrals against the explicit metric solve.
DENSITY_FITTING_TOL = 1e-10
# The erf(ωr)/r kernel with ω = 1e5 against the full 1/r kernel.
ATTENUATION_LIMIT_TOL = 1e-8
# Fitted versus exact four-centre integrals: the accuracy of the universal
# JKFIT auxiliary basis itself, far looser than any kernel tolerance.
FITTED_INTEGRAL_TOL = 1e-2
# Analytic Fourier transforms against trapezoid quadrature of Gaussians on
# a cubic grid, limited by the grid's truncation of the tails.
QUADRATURE_TOL = 1e-9
# Coulomb integrals assembled from Fourier transforms by radial quadrature.
RECIPROCAL_SPACE_TOL = 1e-10
# Density-fitted versus exact Hartree–Fock energies with the universal JKFIT
# auxiliary basis: the fitting error of that basis, not a kernel tolerance.
DENSITY_FITTING_ENERGY_TOL = 1e-4
# Orbital energies of converged states against PySCF; first order in the
# residual orbital error, unlike the variational total energy.
MO_ENERGY_TOL = 1e-7
# Densities of the same state converged in different classes: first order in
# the residual orbital error at the default gradient tolerance.
CONVERGED_DENSITY_TOL = 1e-6
