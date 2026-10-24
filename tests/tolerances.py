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
# The residual of F C = S C ε at a converged result, with the Fock matrix
# rebuilt from the density of the orbitals: first order in the residual
# orbital error at the default gradient tolerance.
FINAL_FOCK_RESIDUAL_TOL = 1e-6
# Spin expectation values of converged states against PySCF: first order in
# the residual orbital error, like the orbital energies.
SPIN_TOL = 1e-7
# Lattice sums cut off at a loose precision against the default cutoff: the
# precision bounds the neglected images up to their Gaussian prefactors.
LATTICE_SUM_LOOSE_PRECISION = 1e-4
LATTICE_SUM_LOOSE_TOL = 1e-3
# Screened lattice sums against the unscreened ones: every neglected term is
# below the threshold, and a few terms per element sit near it.
SCREENING_THRESHOLD = 1e-10
SCREENING_TOL = 1e-9
# The regularized nuclear attraction of a cell against PySCF's density-fitted
# potential at its precision of 1e-14: for diffuse bases (STO-3G lithium) the
# two lattice sums agree to about this level, PySCF's own convergence.
PERIODIC_ATTRACTION_TOL = 1e-10
# The residual of a cell's nuclear attraction in a 40 Bohr box after the
# Madelung term: the quadrupole term (2π/3L³) Σ_A Z_A <μ|(r − A)²|ν> of the
# regularized potential, which the inverse-cube law of the test tracks.
LARGE_BOX_RESIDUAL_TOL = 1e-2
# The ratio of that residual between boxes of 40 and 20 Bohr: 1/8 for the
# inverse-cube law, with room for the next term of the expansion.
LARGE_BOX_RATIO_WINDOW = (0.09, 0.16)
# Fitted periodic integrals against PySCF's Gaussian density fitting with the
# same auxiliary basis and kernel: both converge their lattice sums to about
# this level for diffuse bases (STO-3G lithium).
PERIODIC_FITTING_TOL = 1e-10
# The residual of a cell's fitting integrals in an 80 Bohr box after the
# Madelung terms: the quadrupole term (2π/3L³) of the diffuse universal
# JKFIT functions, which the inverse-cube law of the test tracks.
LARGE_BOX_FITTING_RESIDUAL_TOL = 0.1
# Fitted periodic integrals with a near-singular metric against PySCF: the
# universal JKFIT basis of LiH in a 4 Bohr cell has metric eigenvalues at
# 1e-14 of the largest, and the fit moves by 1e-5 with the eigenvalue
# cutoff, so the two codes agree only to the fitting residual.
ILL_CONDITIONED_FITTING_TOL = 1e-4
# Below this ratio of the smallest to the largest metric eigenvalue the
# auxiliary basis counts as near-linearly dependent.
METRIC_CONDITION_LIMIT = 1e-10
# A slab against a three-dimensional box with vacuum L along its normal:
# after the Madelung term the nuclear attractions differ by the background's
# −(2π/(AL)) Σ_A Z_A <μ|(z − z_A)²|ν> up to terms exponentially small in L,
# so the residual times L agrees between boxes of 20 and 40 Bohr.
LARGE_VACUUM_SLAB_TOL = 1e-9
# A chain against a box with vacuum L × L: after the Madelung term the
# residual is the background's (π/(aL²)) Σ_A Z_A <(r − A)_⊥²> plus a quartic
# term, so it falls off with the inverse square of L; the ratio between
# boxes of 40 and 20 Bohr is 1/4 with room for the quartic term.
LARGE_VACUUM_RESIDUAL_TOL = 0.05
LARGE_VACUUM_RATIO_WINDOW = (0.24, 0.26)
# Positions after the rotation and translation into a standard frame, and
# the frames themselves: double-precision geometry.
GEOMETRY_TOL = 1e-10
# The weight of a symmetry-adapted orbital in its own irrep, from a converged
# or an exactly rotated set of orbitals.
IRREP_WEIGHT_TOL = 1e-10

# A bound on the lattice sum of a function's envelope is loose by the split of
# the exponent that absorbs the polynomial; the largest envelope sampled on a
# grid must still reach this fraction of the bound.
AMPLITUDE_BOUND_RATIO = 0.05
