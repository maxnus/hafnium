# 0037 The kernels differentiate as host callbacks with custom rules over the shell-table arrays

Status: accepted 2026-11-01

## Context

M9 part three makes the compiled kernels differentiable under JAX. PLAN.md
§3.3 and §6.3 foresee the integrals exposed through `jax.ffi` with
`custom_vjp` rules from the derivative integrals, and the functionals of
the `xcx` crate wrapped with `custom_jvp` rules from the derivatives it
supplies. Three things had to be settled: what the differentiable inputs
of an integral are, since a kernel reads a shell table rather than a
molecule; how a kernel that runs on the host enters a traced graph, given
that `jax.ffi` needs an XLA custom call compiled against XLA's C++ headers,
which the Rust crate cannot provide without a C++ shim; and how the
functionals, whose energy per particle the evaluation returns, take a
derivative rule that is well defined where the density vanishes.

## Decision

- **The differentiable inputs are the arrays of the shell table.** A
  `DifferentiableIntegrals` object holds the static part of the orbital
  and auxiliary shell tables, the angular momenta, exponents and
  contraction layout, and the atom of every shell. Each integral type is a
  function of the centres of the shells, ``[shell, 3]``; the centres
  follow the atoms through a gather written in the array namespace, so
  the derivative with respect to the atomic positions is the chain rule
  through it, and nothing in the rules sums over atoms. The exponents and
  contraction coefficients are static for now: a rule must return a
  cotangent for every traced input, and a zero there would be silently
  wrong, so they stay static until their derivative kernels exist, the
  integrals resolved per primitive for the coefficients and the angular
  momentum raised by two for the exponents. The normalization lives in
  Python already and traces when that day comes.
- **A kernel enters the graph as a host callback with a custom
  vector-Jacobian product.** The primal is `jax.pure_callback` around the
  engine, so it traces under `jit` and runs on the host as it does today.
  The backward pass is another callback that evaluates the derivative
  kernels of ADR-0035 and contracts them with the cotangent on the host:
  the bra derivative of a symmetric integral is paired with the
  symmetrized cotangent, which supplies the ket term, the per-charge
  derivative of the nuclear attraction with the cotangent itself, and the
  rows are summed into shells. The contraction on the host keeps the
  derivative integrals, which are three times the size of the integrals,
  off the device and out of the graph. `jax.ffi` remains the upgrade for
  when the kernels should run inside the compiled program, which is M10's
  concern.
- **The functional evaluation differentiates through its energy density.**
  Under JAX `evaluate_functional` is a host callback returning the energy
  density ``ρ ε_xc`` and the first derivatives, with a custom
  Jacobian-vector product from the second derivatives that `xcx` supplies
  in libxc's packed layouts, bound as `evaluate_functional_second`. The
  energy per particle is recovered by a division guarded where the total
  density vanishes, where the kernel returns zero. Under NumPy the
  evaluation is unchanged and returns the kernel's energy per particle.
- **The rules live beside their methods and load with JAX.** The JAX-only
  modules `integrals/_jax_rules.py` and `xc/_jax_rules.py` import JAX and
  are imported when a JAX array arrives, so JAX stays optional, and the
  dispatch is a type check on the argument, not a global switch.

## Consequences

- Under JAX the five integral types, the nuclear repulsion and the
  functional evaluation trace under `jit` and differentiate under `grad`
  and `jvp`; the gradients equal the contracted derivative kernels to
  1e-12 and finite differences to 1e-7, and the gradient of the
  Hartree–Fock energy at fixed density equals the analytic gradient of
  ADR-0035 without its overlap term, which is the orbital response that
  part four adds through implicit differentiation.
- The functional's Jacobian-vector products match finite differences of
  the first derivatives for LDA, GGA and meta-GGA functionals, and the
  gradient of the integrated energy density is the potential.
- Every call of a differentiated kernel runs on the host, under `jit` as
  well: the graph pauses at each callback. That is the cost of not having
  an XLA custom call, and it is the same host work the kernels do today.
- Under any backend other than JAX the same methods evaluate the kernels
  directly and return arrays of that backend, so code written against
  `DifferentiableIntegrals` runs everywhere.
