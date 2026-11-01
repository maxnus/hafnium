# 0035 Derivative integrals come from the undifferentiated kernels, and the gradient contracts them at fixed density

Status: accepted 2026-11-01

## Context

M9 opens with the derivatives of the integrals with respect to the
positions of the nuclei: the vector-Jacobian products of the integral
kernels under JAX (M9), the analytic gradients of M11 and the
kinetic-balance integrals of M12 all need them, and the native engine
(ADR-0008) had none. Four things had to be settled: how a derivative
with respect to the centre of a basis function is formed in the
McMurchie–Davidson scheme; how the derivative with respect to the
position of a point charge, which is not the centre of a function, is
formed; which derivatives a kernel returns, since translational
invariance ties the derivatives of an integral together; and how the
gradient of the energy is assembled from them for all eight Fukutome
classes, with or without point-group sectors.

## Decision

- **A derivative with respect to the centre of a function in a real
  pair raises and lowers its angular momentum.** For a Cartesian
  primitive ``∂_A G_a = 2α G_{a + e} − a_e G_{a − e}`` along each axis
  ``e``, so the derivative kernels evaluate the value function of the
  undifferentiated kernel for the components of ``l + 1`` and ``l − 1``
  from one Hermite expansion built with an extra order on the bra, and
  combine them per primitive, where the exponent is known. The overlap,
  the kinetic energy, the nuclear attraction and the orbital side of the
  three-centre integrals share this path with their undifferentiated
  kernels, whose value functions are the same code.
- **A derivative with respect to a centre that is not the centre of a
  product shifts the Hermite Coulomb integrals.** For a point charge
  ``∂_C R_{tuv}(p, P − C) = −R_{t+1,uv}``, and for a function with a unit
  partner (an auxiliary function) the Hermite centre is its own centre,
  so ``∂_P R_{tuv}(α, P − Q) = R_{t+1,uv}`` and ``∂_Q`` the negative. The
  nuclear attraction, the two-centre metric and the auxiliary side of the
  three-centre integrals take this path, which costs one more order of
  the Hermite table and nothing else.
- **Each kernel returns the derivatives its translational invariance does
  not fix.** The overlap, kinetic and two-centre kernels return the bra
  derivative, ``[image, 3, bra, ket]``, the ket derivative being its
  negative; the attraction returns the bra derivative and the derivative
  with respect to every charge, ``[image, charge, 3, bra, ket]``; the
  three-centre kernel returns the auxiliary and the bra derivatives, each
  ``[image, 3, aux, bra, ket]``. A molecule's ket derivatives are the
  transposes of the bra ones for the symmetric operators, and the
  engine protocol carries the same signatures as the undifferentiated
  methods with a leading axis for the component. The derivative kernels
  evaluate every term: the screening bounds of ADR-0018 are not derived
  for them, since the gradients of molecules do not screen, and the
  periodic gradients that would will bring the bounds with the
  Bloch-summed derivatives.
- **The gradient is assembled in the dense atomic basis at fixed
  density.** The Hartree–Fock energy is stationary in the orbitals, so
  its gradient is the derivative of the integrals paired with the
  converged densities plus the overlap term of the energy-weighted
  density ``W = D F D``, which the converged density and Fock matrix give
  for every class without the orbitals. The spin-summed density pairs
  with the one-electron and the Coulomb terms, the spin-orbital density
  ``[2n, 2n]`` of :meth:`BlockMatrix.to_dense` with the exchange term, so
  that one expression covers the restricted, unrestricted and
  generalized classes, real, complex and paired alike, and the irrep
  sectors are resolved by the dense forms. With density fitting the
  three-centre derivatives pair with an effective density per auxiliary
  function, ``c_P D − Σ_Q (M⁻¹)_PQ T_Q`` for the fitted charges ``c`` and
  the exchange intermediates ``T_Q = Σ_s (D L̃_Q D)_ss``, and the metric
  derivative with ``½ (M⁻¹ W M⁻¹ − c cᵀ)`` for ``W_PQ = tr(L_P D L_Q D)``;
  the atom of every function sums the centres. The result reports the
  terms, :class:`NuclearGradient`, like :class:`EnergyTerms` reports the
  energy. The gradient covers density-fitted Hartree–Fock: the exact
  four-centre derivatives are not formed, and a Kohn–Sham result is
  rejected until the derivative of the functional term arrives with M11.
- **Every derivative is tested against finite differences, and the
  conventions against libcint.** In Rust each kernel is differenced with
  respect to the centre of every shell and every charge; in Python the
  integrals of a molecule are differenced under the move of every atom,
  and the gradient against five-point differences of the energy, one
  difference per molecule and every class against it. The reference tests
  compare with PySCF's ``ip`` integrals, which differentiate with respect
  to the electron coordinate and so carry the opposite sign, and with its
  density-fitted RHF and UHF gradients.

## Consequences

- A derivative component costs about what the undifferentiated integral
  costs; the centre derivative evaluates two component sets per
  primitive pair and the Hermite shift one more order of the table.
- The per-nucleus attraction derivative, ``[atom, 3, ao, ao]``, and the
  two three-centre derivatives, ``[3, aux, ao, ao]`` each, are held in
  full and contracted in Python. Contracting them with the effective
  densities inside the kernels, which the vector-Jacobian products of M9
  also want, is the optimization when larger molecules need it.
- Hartree–Fock with exact integrals has no gradient, since the
  four-centre derivatives are not formed; Kohn–Sham gradients need the
  derivative of the exchange–correlation term and the grid weights (M11);
  periodic gradients need the Bloch-summed derivatives with their
  screening bounds.
- The JAX bindings of M9 register these kernels as the vector-Jacobian
  products of the integrals; the effective densities of the gradient are
  the cotangents they receive.
