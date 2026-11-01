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

# Lebedev rules against the closed-form integrals of monomials up to their
# degree, and the sum of their weights against one.
ANGULAR_QUADRATURE_TOL = 1e-13
# Radial quadratures against the closed-form integrals of Gaussians, relative.
RADIAL_QUADRATURE_TOL = 1e-11
# Becke's partition weights summed over the atoms against one.
PARTITION_TOL = 1e-13
# Basis functions on grid points against a direct evaluation of the
# contracted solid harmonics.
BASIS_VALUE_TOL = 1e-12
# Analytic derivatives against central finite differences with a step of 1e-5.
FINITE_DIFFERENCE_TOL = 1e-7
# Basis values left out beyond the cutoff radii of this precision.
BASIS_CUTOFF_PRECISION = 1e-10
# Functional values and derivatives against PySCF's libxc at the same points,
# relative: the two implementations differ in rounding only.
LIBXC_REL_TOL = 1e-12
# Kohn–Sham energies against PySCF with grids built from the same tables: the
# grids coincide up to the CODATA value of the Bohr, which moves the weights
# of the steepest partition regions by 1e-9 relative, and both codes converge
# to 1e-12.
KOHN_SHAM_ENERGY_TOL = 1e-9
# Grid points against PySCF's: the positions differ by the CODATA value of
# the Bohr the two codes convert Ångström with.
GRID_POINT_TOL = 1e-9
# Grid weights against PySCF's, relative, for weights above the floor; the
# tiny weights far from the atoms are the most sensitive to the positions.
GRID_WEIGHT_REL_TOL = 1e-8
GRID_WEIGHT_FLOOR = 1e-6
# The orbital gradient at which the Kohn–Sham states compared with PySCF are
# converged, PySCF's conv_tol_grad there, so that the orbital energies compare
# at MO_ENERGY_TOL.
REFERENCE_GRADIENT_TOL = 1e-8

# The quadrature of a cell's grid (ADR-0027) on Bloch sums of Gaussians and on
# the electron density at level 3, relative: the shared-shell rule brings the
# periodic partition to this accuracy for the test cells.
PERIODIC_GRID_TOL = 1e-5
# Kohn–Sham energies of a cell on Hafnium's grid against PySCF's on its own
# periodic Becke grid at the same level: PySCF truncates the atomic grids at
# the cell faces, so the two quadratures differ by this much; on one and the
# same grid the two codes agree at KOHN_SHAM_ENERGY_TOL.
PERIODIC_GRID_ENERGY_TOL = 1e-4
# Kohn–Sham energies of a cell at grid levels 3 and 5.
PERIODIC_LEVEL_TOL = 1e-5
# The exchange–correlation term of a molecule in a box of 80 Bohr at the Γ
# point against the molecular term on the same grid: the partition is shared
# with the images only where the density has decayed.
MOLECULAR_LIMIT_TOL = 1e-9

# A GGA on the sphere of spin directions is invariant under spin rotations
# only to the quadrature error of the Lebedev rule of 770 points, which
# carries the octahedral group alone (ADR-0029); at a polarization of 0.5
# the error of the energy density stays below this.
SPIN_ROTATION_QUADRATURE_TOL = 1e-9

# A generalized Kohn–Sham state with a collinear density reproduces the
# restricted or unrestricted energy: exactly for an LDA, and for a GGA or
# meta-GGA to the quadrature error of the sphere of spin directions at the
# points polarized below the collinear threshold (ADR-0029).
NONCOLLINEAR_LIMIT_TOL = 1e-8
# Second-order energies against PySCF and across the Fukutome classes: the
# correlation energy is first order in the orbital error, so the references
# are converged to a gradient of 1e-9 on both sides before comparing.
MP2_ENERGY_TOL = 1e-10
# CCSD energies against PySCF, across the Fukutome classes, and against
# full configuration interaction for two electrons: both sides converge the
# reference to a gradient of 1e-9 and the amplitudes to 1e-10.
CCSD_ENERGY_TOL = 1e-10

# Natural occupations of CCSD densities, both codes' amplitudes converged to
# a norm of 1e-10.
NATURAL_OCCUPATION_TOL = 1e-8

# Analytic nuclear gradients against five-point central differences of the
# energy with a step of 2e-3 Bohr, the states converged to a gradient of 1e-10:
# the differences are limited by the rounding of the energies.
GRADIENT_FINITE_DIFFERENCE_TOL = 1e-7
# Analytic nuclear gradients against PySCF's density-fitted gradients with the
# same auxiliary basis, and across the Fukutome classes: first order in the
# residual orbital error.
GRADIENT_TOL = 1e-7
