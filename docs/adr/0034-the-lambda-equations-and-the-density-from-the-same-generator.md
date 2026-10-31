# 0034 The Λ equations and the one-particle density come from the same generator as the linked terms of the energy functional

Status: accepted 2026-10-31

## Context

M8 lists the Λ equations after (T): the de-excitation operators
``Λ = Λ1 + Λ2`` that make the energy functional
``<Φ| (1 + Λ) e^{-T} H_N e^T |Φ>`` stationary in the amplitudes, which
every property of the CCSD state and later its gradients need through
the reduced densities. Five things had to be settled: how the terms are
derived, since the Λ equations are not connected in the sense of
ADR-0032 (the ``λ_i^a F_jb`` term of the doubles joins the Hamiltonian
and ``Λ`` only through the ket); what structure the de-excitation
amplitudes have and where their iteration starts; how the one-particle
density is formed and reported; how the longer generated functions,
whose variables the type checker stops following, are typed; and how
the result is exposed.

## Decision

- **The generator derives the Λ equations as the linked terms of
  ``<Φ| (1 + Λ) (H_N e^T)_C |Φ_μ>``.** ``Λ1 = Σ l1[i, a] a†_i a_a`` and
  ``Λ2 = ¼ Σ l2[i, j, a, b] a†_i a†_j a_b a_a`` are operator strings of
  two new tensor kinds, the ket of a projection is the adjoint of its
  bra, and the expectation value keeps the terms in which every cluster
  operator is contracted with the Hamiltonian and drops those in which
  the Hamiltonian with its clusters closes on itself, contracted with
  neither ``Λ`` nor the ket: the unlinked products of a de-excitation
  amplitude with the energy, which the ``E`` of the functional cancels.
  A term in which ``Λ`` and the Hamiltonian touch only through the ket
  stays. The singles equation has 59 terms, the doubles 69, with the
  linear terms of the textbook (``f_ia``, ``λ_i^e f_ea − λ_m^a f_im``,
  ``λ_m^e <ie||am>``, ``½ λ_im^ef <ef||am>``, ``−½ λ_mn^ae <ie||mn>``,
  ``<ij||ab>``, the ladders and ``P(ij) P(ab) λ_i^a f_jb``); the
  derivation takes a second.
- **The de-excitation amplitudes have the structure of the conjugate
  amplitudes and start there.** ``l1[i, a]`` and ``l2[i, j, a, b]``
  conjugate the occupied indices, the index character of
  ``a†_i a†_j a_b a_a`` (CONVENTIONS.md §3), so that the residuals the
  program produces share their structure and the same update ``R / Δ``
  applies; ``conj(t)`` solves the equations to first order, since
  ``λ_ij^ab ≈ <ij||ab> / Δ``, and is the starting point. The iteration
  is that of the amplitudes with the DIIS of ADR-0032 and the tolerances
  of `CcsdOptions`, the amplitude tolerance deciding since the equations
  are linear in ``Λ`` and have no energy; it converges in as many
  iterations as CCSD.
- **The density comes from the same generator and is reported like the
  density of the mean field.** The blocks
  ``<Φ| (1 + Λ) ({p† q} e^T)_C |Φ>`` by the spaces of ``p`` and ``q`` are
  equations without antisymmetry, one orbit per term, written as one
  function returning the mapping; ``D[p, q] = <a†_q a_p>`` is the
  occupations of the reference plus the transposed blocks, and its
  Hermitian part is taken, as PySCF does, since the density of a
  non-variational theory is not Hermitian. `OrbitalSpaces.to_atomic_basis`
  forms ``C X C†`` per stored block and point of the structure of the
  state, so the density is a `BlockMatrix` like `ScfResult.density`, with
  `natural_orbitals` as the inverse of `density_matrix`. The two-particle
  density waits for the gradients.
- **Every generated function declares its variables first.** Pyright
  infers the type of an undeclared variable from its assignments and
  gives up beyond 64 of them, which the singles residual of the Λ
  equations exceeds with its 204 statements; a bare annotation
  ``x: BlockTensor`` of every variable a function assigns, written
  before the first statement, declares the type once and leaves the
  number of assignments free, so the three generated modules open their
  functions that way rather than splitting the longer equations into
  parts.
- **The result is a subclass with its own entry points.**
  `CcsdLambdaResult` extends `CcsdResult` by the de-excitation
  amplitudes, the density and the Λ iteration's history; `solve_lambda`
  takes an existing CCSD result, and `df_ccsd_lambda`,
  `ccsd_lambda_from_df_tensor`, `ccsd_lambda_from_periodic_fitting` and
  `solve_ccsd_lambda` mirror the CCSD entry points.

## Consequences

- One Λ for the eight classes, the point-group sectors and k-points: the
  densities agree across the classes and with the plain reference of a
  blocked one, their trace with the overlap is the electron count, the
  density is the derivative of the correlation energy with respect to
  the orbital energies by finite differences, equals the full-CI density
  for two electrons, folds onto the supercell at Γ, and its natural
  occupations match PySCF's after ``solve_lambda`` and ``make_rdm1`` for
  RHF, UHF and GHF references to 1e-10. PySCF has no Λ over k-points, so
  that case is checked against the classes and the supercell alone.
- On 16 cores the Λ step of water in def2-SVP takes 0.24 s (restricted),
  0.32 s (unrestricted) and 0.52 s (generalized) against PySCF's
  ``solve_lambda`` at 0.58 s, 1.99 s and 0.47 s, with the CCSD step of
  the same run at 0.19 s, 0.26 s and 0.44 s against 0.86 s, 2.32 s and
  1.44 s.
- The Λ equations read the blocks of the Hamiltonian the CCSD equations
  do but ``<ab||ij>``, among them the ``<ab||cd>`` ladder of
  ``½ λ_ij^ef <ef||ab>``; the de-excitation amplitudes double the
  memory of the amplitudes.
- Later: the two-particle density and the Λ of (T) for the densities
  and gradients of CCSD(T), properties from the density (dipole moments
  once the multipole integrals exist), and relaxed densities with the
  orbital response.
