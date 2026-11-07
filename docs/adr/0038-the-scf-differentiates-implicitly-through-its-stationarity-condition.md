# 0038 The SCF differentiates implicitly through its stationarity condition

Status: accepted 2026-11-07

## Context

M9 part four makes the converged mean-field state a differentiable function
of the nuclear positions, so that `jax.grad` of any function of the state
is its derivative, and closes the milestone's exit criterion for
Hartree–Fock: the AD gradient equals the analytic gradient of ADR-0035 and
finite differences of the energy. PLAN.md §6.2 rules out unrolling the
iterations and asks for implicit differentiation at the fixed point, the
coupled-perturbed equations in disguise. Four things had to be settled:
in which variables the fixed point is written, since a fixed point in the
orbital coefficients has the gauge freedom of rotations within the
occupied and within the virtual orbitals and a fixed point through the
eigendecomposition of the Fock matrix differentiates through degenerate
eigenvalues; how the orbitals follow the positions, since the converged
coefficients are orthonormal in the overlap of the positions they were
solved at and nowhere else; how the response equations are solved without
writing them out for each of the eight Fukutome classes; and how the
structured algebra, whose constructors check what they are given, runs on
abstract values.

## Decision

- **The stationarity condition is the orbital gradient in the
  occupied–virtual rotations.** With the converged orbitals ``C`` of a
  block and a rotation ``κ`` of shape ``[virt, occ]``, the orbitals
  ``C (1 + K)`` with ``K = [[0, −κ†], [κ, 0]]`` are the first order of a
  unitary rotation, and the residual ``r(κ, R) = C_v† F(D) C_o`` of the
  rotated orbitals vanishes at the solution. Its Jacobian in ``κ`` is the
  electronic Hessian up to the occupation numbers, self-adjoint and
  positive definite at a stable state, and it has no components along the
  gauge, so the implicit function theorem applies to it directly. Only the
  first order of the rotation enters a derivative at zero rotation, so the
  exponential is not formed. The occupations are those of the reference
  solve and must fill the lowest orbitals of every block equally, which
  the aufbau filling of a state away from a level crossing does.
- **The orbitals follow the positions through a Cholesky
  re-orthonormalization.** The connection ``C L^{-†}`` with the factor
  ``C† S(R) C = L L†`` is orthonormal in the overlap at ``R`` and is ``C``
  itself at the positions of the solve. Any smooth re-orthonormalization
  gives the same derivative of a stationary energy, and this one carries
  the term of the energy-weighted density against the derivative of the
  overlap, which the fixed-density gradient of ADR-0037 lacked, without an
  eigendecomposition to differentiate through.
- **The solver is a host callback with a custom vector-Jacobian
  product.** The forward pass runs the SCF of `solve_scf` on the host,
  from the atomic density guess and with the convergence controls of
  `ScfOptions`, as a `jax.pure_callback` that returns the orbitals. The
  backward pass pulls a cotangent on the orbitals back to the positions
  through the connection and through the rotations: the latter cotangent
  is the right-hand side of the transposed coupled-perturbed equations,
  solved by conjugate gradients with the vector-Jacobian products of the
  residual in the rotations, one Fock build each, preconditioned by the
  orbital-energy differences; the solution is then pulled back through
  the residual's own dependence on the integrals. Every transpose is
  JAX's own, so the complex classes follow its conventions and the
  operator stays self-adjoint in the real inner product the solver uses.
  The cotangents of the integrals from the connection and from the
  response are combined before the derivative integrals of ADR-0035 are
  evaluated, once. A stationary function of the state, the energy, has a
  vanishing right-hand side and costs no response; an orbital energy or an
  expectation value gets the full response.
- **The two-electron term applies the Cholesky factor of the metric.** The
  differentiable Hamiltonian fits the three-centre integrals with
  ``L⁻¹ (P|uv)`` for ``(P|Q) = L L†``, whose products reproduce the fitted
  four-centre integrals as the absorbed ``M^{-1/2}`` of ADR-0007 does,
  without the eigenvalue threshold that an eigendecomposition would need to
  differentiate through; the host solve of the forward pass uses the same
  Hamiltonian, so the fixed point and the differentiated functional agree.
  The energy terms of the Fock build are 0-d arrays of the backend, read
  with `float()` at the host boundary, so that the energy traces.
- **Validations that read entries pass abstract values.** The closeness
  check of `from_spatial` and the time-reversal check of `density_matrix`
  run on concrete arrays only; a tracer under a JAX transformation has no
  entries to read, and the concrete arrays it stands for were checked when
  they were built. Everything else in the structured algebra is shapes and
  dtypes, which trace.
- **The scope is density-fitted Hartree–Fock of a molecule.** A Kohn–Sham
  energy needs the derivatives of the basis functions on the grid with
  respect to their centres, up to second order for the gradient of a
  meta-GGA, which the grid kernel does not provide; a cell needs the
  derivative kernels under the lattice sums. Both reuse the solver's rules
  unchanged and are later parts.

## Consequences

- `DifferentiableScf.solve` returns the state at any positions as arrays
  of the active backend, and under JAX `jax.grad` of its energy equals the
  analytic gradient to 1e-7 in all eight classes and with point-group
  sectors, and finite differences of the energy to 1e-7; the derivative of
  an orbital energy, which needs the response, equals finite differences
  of converged states to 1e-7, and `jit` of the gradient agrees with the
  eager evaluation to 1e-12.
- A derivative with respect to any other input of the integrals, the
  exponents and contraction coefficients of the basis once their kernels
  exist, goes through the same rules without change: the integrals are
  the only place the positions enter.
- Derivatives of the orbital coefficients themselves are taken in the
  gauge of the connection and the rotations; the density, the energy and
  the orbital energies to first order are gauge-independent.
- The response solve is checked for convergence on concrete arrays only,
  since under `jit` the residual cannot be read; a state with fractional
  occupations, or whose basis the overlap threshold truncates, is
  rejected at construction.
