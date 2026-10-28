# 0029 Noncollinear functionals by the multi-collinear approach

Status: accepted 2026-10-25

## Context

Kohn–Sham for the generalized classes (real, complex and paired GHF, the
TSDW, TSW and TSCW of Fukutome) needs an exchange–correlation functional
of the 2×2 spin-density matrix ``ρ_στ(r)``, or equivalently of the density
``ρ`` and the magnetization ``m = (2 Re ρ_αβ, −2 Im ρ_αβ, ρ_αα − ρ_ββ)``
with their gradients and the kinetic energy densities ``τ`` and ``τ_m``.
The functionals of libxc, which ``xcx`` reproduces (ADR-0026), are
collinear: they read ``ρ_α`` and ``ρ_β``, so a rule is needed that turns a
collinear functional into one of ``(ρ, m)``. Two are in use:

- The *locally collinear* approach evaluates the collinear functional at
  the spin densities ``(ρ ± |m|)/2`` at every point, the local spin axis
  being that of ``m``. For an LDA this is the natural and complete answer.
  For a GGA the spin gradient must be chosen: ``∇|m|``, or the gradient of
  the projection on a fixed axis, or the per-component norms; the choices
  differ wherever the direction of ``m`` varies in space, and none is
  distinguished.
- The *multi-collinear* approach of Pu, Zhang, Xu and co-workers (Phys.
  Rev. Research 5, 013036, 2023) projects every spin variable
  ``S = (m, ∇m, τ_m)`` onto a direction ``n`` of spin space, ``s_n = n·S``,
  evaluates the collinear functional ``f`` at ``(ρ ± s_n)/2`` and averages
  over the sphere with a derivative term,

  ```
  f_mc(ρ, S) = ∫ dΩ/4π [ f(ρ, n·S) + (n·S)·∂f/∂s (ρ, n·S) ]
             = ∫ dΩ/4π d/dt [ t f(ρ, t n·S) ] at t = 1.
  ```

  Where the spin variables are parallel (a collinear density) the
  integrand depends on ``u = n·m̂`` alone and
  ``½ ∫_{−1}^{1} d/du [u f(ρ, u s)] du = ½ [f(ρ, s) + f(ρ, −s)] = f(ρ, s)``,
  so the spin-polarized functional is recovered exactly at every rung,
  which the locally collinear GGA variants do not all manage. The
  functional is unique, invariant under rotations in spin space, and its
  potential needs the second derivatives of ``f``. PySCF implements it in
  GKS through the ``mcfun`` package (``collinear='mcol'``).

For an LDA the two coincide: the sphere integral has the closed form
``f(ρ, |m|)``.

The integrand is smooth as long as both projected spin densities stay
away from zero. Where ``|m|`` approaches ``ρ`` the point of the sphere with
``n = m̂`` has ``ρ_β → 0``, where the functionals have branch points
(``ρ_β^{4/3}`` in the exchange, logarithms in the correlation, reduced
gradients ``|∇ρ_β|/ρ_β^{4/3}`` that diverge) and a quadrature rule of
degree ``d`` converges like ``r^{−d}`` with ``r = 1/ζ + √(1/ζ² − 1)`` for
the polarization ``ζ = |m|/ρ``: at ``ζ = 0.9`` PySCF's Lebedev rule of 770
points (degree 47) is exact to 1e-10, at ``ζ = 0.99`` only to 1e-3, and a
fully polarized point (a one-electron region) is not integrable to any
useful precision. PySCF therefore switches such points to a collinear
evaluation. Large spin gradients bring singularities of their own, so
the polarization alone does not predict the error of a rule.

The second derivatives are the cost: ``xcx`` differentiates forward, and
a GGA second derivative costs about five first derivatives, a meta-GGA
one about fifteen. One fixed rule of 770 directions at every point makes
a PBE Kohn–Sham term thirty times and a TPSS term several hundred times
as expensive as the collinear one, while most points, where the spin
variables are small against the total ones, would be integrated to
rounding by a handful of directions.

## Decision

The generalized classes take the multi-collinear form of the functional,
evaluated in the compiled layer by ``xc::evaluate_noncollinear`` and
wrapped as :func:`hafnium.xc.evaluate_noncollinear` on a
:class:`~hafnium.xc.NoncollinearDensity`:

- An LDA takes the closed form ``f(ρ, |m|)`` at every point, with
  ``∂f_mc/∂m = ½ (v_α − v_β) m̂``; no sphere is sampled.
- A GGA or meta-GGA samples the sphere. At every direction the kernel
  evaluates ``f`` with its second derivatives through ``xcx`` and
  accumulates the energy density ``f + S·∂f/∂S`` and the potential
  ``∂f/∂T + S·∂²f/∂S∂T`` for the total variables ``T = (ρ, ∇ρ, τ)`` and
  ``n (2 ∂f/∂S + S·∂²f/∂S²)`` for the spin variables, the chain rule from
  libxc's variables ``(ρ_σ, σ_σσ', τ_σ)`` to the total and spin variables
  written out once.
- The directions are, by default, product rules aligned with the
  magnetization of each point: Gauss–Legendre nodes in the cosine of the
  angle to ``m``, where the integrand varies most, times as many uniform
  azimuths about it, ``2·2^level`` of each. A point starts at level 0
  (4 directions) and is refined until the energy density and the
  first-order energy response ``Σ_x |∂f/∂x| |x|`` over all variables
  change by less than a precision between two levels, 1e-14 Hartree per
  point by default once the point's quadrature weight is applied, so
  that the energy integrates to within the number of points times the
  precision and the potential stays consistent with it. Points with
  small spin variables converge at the first comparison; strongly
  polarized ones climb to a few thousand directions; the finest level
  (128 × 128 by default, 256 × 256 at most) is reported when it does not
  suffice (:class:`~hafnium.xc.NoncollinearOptions`). The alignment
  makes the energy invariant under rotations in spin space to the
  precision, and the collinear limit, whose integrand depends on the
  cosine alone, exact to the precision at every polarization the rules
  reach.
- A Lebedev rule of a chosen size for every point remains available
  (``lebedev_points``), which is PySCF's scheme with 770 points; the
  rules of :mod:`hafnium.grid` coincide with ``mcfun``'s point by point,
  so the two codes can be compared to rounding.
- Points with ``|m| ≥ θ ρ``, ``θ = 0.99`` by default and ``None`` to
  switch the branch off, take the locally collinear functional with the
  spin variables ``s = |m|``, ``∇s = ∇|m| = m̂·∇m`` and ``τ_s = m̂·τ_m`` and
  its exact derivatives, which carry the terms from the rotation of
  ``m̂`` with ``m``. This is the limit no rule reaches for a collinear
  density and a variational functional for a noncollinear one. It differs from PySCF's branch, which takes the norm of each spin
  variable on its own and the direction of ``m`` for the whole
  potential, so that its potential is not the derivative of its energy;
  the two agree for collinear densities.
- Points whose spin variables are all below ``SPIN_NEGLECT = 1e-14`` of
  the total ones are evaluated unpolarized, ``f(ρ, 0)`` with a zero spin
  potential, which the multi-collinear energy matches to the second order
  in the spin variables. The paired class, whose magnetization vanishes
  by time-reversal symmetry, and a closed shell in the real or complex
  class cost no more than the restricted one.

The exchange–correlation term (:mod:`hafnium.scf.exchange_correlation`)
evaluates a spinor density through ``ρ_στ = Σ_k w_k Σ_uv φ_v^k D[vσ, uτ](k)
conj(φ_u^k)`` and its derivatives on the grid, forms ``ρ``, ``m`` and
their gradients and kinetic energy densities, and assembles the potential
as ``∫ conj(φ_u) (v_ρ δ_στ + v_m·σ_στ) φ_v`` with the Pauli matrices, the
same combinations of the gradient derivatives on ``∂_i(conj(φ_u) φ_v)``
and of the kinetic ones on ``½ ∂conj(φ_u)·∂φ_v``: the alpha–alpha block
carries ``v + v_z``, the beta–beta block ``v − v_z``, the alpha–beta block
``v_x − i v_y`` and the beta–alpha block its adjoint. A real structure
has ``m_y = 0`` and a real potential, the rule being symmetric under
``n_y → −n_y``; a cell runs over the full k set and reduces onto the
stored points as the Coulomb matrix does (ADR-0020, ADR-0027). Hybrids
take their exact exchange from the spinor exchange matrix of the
Hamiltonian as GHF does.

## Consequences

- Kohn–Sham DFT serves all eight Fukutome classes, for molecules and
  cells, with every rung and hybrids. In the collinear limit the
  generalized classes reproduce the restricted and unrestricted energies
  to 1e-13 Hartree for the water cation at every rung: exactly for an
  LDA and at strongly polarized points, to the precision of the aligned
  rules elsewhere.
- The kernel reproduces ``mcfun`` to rounding (1e-14 in the energy
  density and potentials at random noncollinear points for SVWN, PBE and
  TPSS) on the same Lebedev rule with the collinear branch off, and the
  term reproduces PySCF's energy and spinor potential of a noncollinear
  density on PySCF's grid; PySCF's GKS energies of collinear states
  match to the convergence tolerance with the default settings. A
  noncollinear state compares to PySCF only on the same rule with the
  branch off, since the branches differ.
- The cost of a GGA or meta-GGA term grows with the number of directions
  the points need. For the water cation in STO-3G on the default grid
  (16 threads) a PBE Kohn–Sham run takes 6.7 s in the complex
  generalized class against 0.3 s in the unrestricted one, where the
  fixed rule of 770 directions took 26 s; a precision of 1e-12 brings it
  to 4.6 s and 1e-11 to 3.6 s, the energy unchanged to 1e-13 in this
  collinear case. TPSS takes 105 s against 0.3 s: its second derivatives
  cost fifteen first derivatives in ``xcx`` and the kinetic variables
  demand finer rules. Closed shells and the paired class skip the sphere
  and cost nothing extra; nor does an LDA (0.3 s against 0.1 s).
- A GGA or meta-GGA energy is invariant under rotations in spin space to
  the precision of the aligned rules, which turn with the magnetization;
  a Lebedev rule carries the octahedral group alone and is invariant
  only to its quadrature error. The LDA is exactly invariant.
- The functional is piecewise at the threshold: the sphere and the
  locally collinear branch agree for collinear densities and differ by
  the noncollinear part of the gradients where a point crosses
  ``|m| = θ ρ``. A smooth blend between the two branches, a cheaper
  second derivative (a directional one in place of the full Hessian
  ``xcx`` returns) and the second derivatives of ``f_mc`` for response
  theory are later items.
