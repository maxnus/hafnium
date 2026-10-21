"""Named tolerances for tests (CONVENTIONS.md §8). Never use inline literals."""

ENERGY_TOL = 1e-10
INTEGRAL_TOL = 1e-12
GRID_TOL = 1e-6
# Kernel results versus NumPy: the two sum in different orders (sequential
# loop versus BLAS), so they agree only to about n * machine epsilon.
KERNEL_REL_TOL = 1e-12
