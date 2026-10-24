# Bundled grid data

`lebedev.json` holds the Lebedev–Laikov quadrature rules on the unit sphere
in their compact form: for every rule the number of points, the algebraic
degree of the polynomials it integrates exactly, and the orbit generators
`(code, a, b, weight)` from which the points follow by the octahedral
symmetry (`hafnium.grid.angular`). The weights of a rule sum to one.

Source: V. I. Lebedev and D. N. Laikov, Doklady Mathematics 59, 477 (1999);
the parameters are the published values as distributed by Dmitri Laikov
through CCL and tabulated in PySCF 2.14 (`pyscf/dft/LebedevGrid.py`, from
Gerald Knizia's `CxLebedevGrid`), Apache-2.0. Please cite the paper when
using the rules in published work.
