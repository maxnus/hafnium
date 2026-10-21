# ADR-0008: McMurchie–Davidson for the native integral engine

Status: accepted, 2026-10-21.

## Context

The compiled layer evaluates Gaussian integrals natively after a libcint
bootstrap (ADR-0002). The integral types it must cover (PLAN.md §3.4) are
one-electron integrals (overlap, kinetic, nuclear attraction, multipoles),
two- and three-centre Coulomb integrals for density fitting, range-separated
variants of every Coulomb-type integral, the Fourier transform of
basis-function pairs and of auxiliary functions for periodic systems
(ADR-0007), four-centre electron repulsion integrals, and first derivatives
of all of these. Density fitting is the primary route to two-electron
terms; exact four-centre integrals are secondary.

Candidates: McMurchie–Davidson (Hermite Gaussian expansion of each
overlap distribution, Coulomb terms from Hermite integrals of the Boys
function), Rys quadrature (libcint's scheme for electron repulsion
integrals), and Obara–Saika with Head-Gordon–Pople transfer (libint's
scheme). The criteria were coverage of every integral type from shared
intermediates, simplicity of the numerical core, performance of
four-centre integrals at high angular momentum, and code size.

## Decision

The native engine uses the McMurchie–Davidson scheme for every integral
type.

- **Shell-pair intermediates.** For each pair of primitive shells, and in
  periodic systems for each lattice image of the pair, the engine computes
  the product exponent `p`, the product centre `P`, the prefactor and the
  Hermite expansion coefficients `E_t` along each Cartesian axis by the
  standard two-term recursion, stores them in structure-of-arrays form, and
  reuses them for every integral over that pair.
- **Coulomb-type integrals** come from the Hermite Coulomb integrals
  `R_{tuv}` built from the Boys function `F_n(T)` by the usual recursion.
  Range separation with the erf-attenuated kernel replaces the exponent
  `α` of the charge distribution seen by the kernel (`p` for nuclear
  attraction, the reduced exponent `pq / (p + q)` for two-electron
  integrals) by `α ω² / (α + ω²)` wherever it appears: in the Boys
  argument, in the `(−2α)^n` factors of the Hermite recursion and in the
  `√α` prefactor, so the order-`n` auxiliary integral is scaled by
  `(ω² / (α + ω²))^(n + 1/2)`. The erfc kernel is the difference from the
  full kernel.
- **Pair Fourier transforms** are analytic in the Hermite basis: the
  transform of the Hermite Gaussian `Λ_{tuv}` with kernel `exp(−i G·r)` is
  `(π/p)^{3/2} exp(−G²/4p) exp(−i G·P) (−i G_x)^t (−i G_y)^u (−i G_z)^v`,
  so the reciprocal-space terms of ADR-0007 reuse the shell-pair
  coefficients.
- **Derivative integrals** use the derivative of a Cartesian Gaussian as a
  combination of the functions with angular momentum raised and lowered by
  one, so they reuse the expansion coefficients of the neighbouring
  angular momenta.
- **Angular momentum.** Integrals are evaluated over Cartesian Gaussians
  and transformed to real solid harmonics per shell with the ordering and
  phase of ADR-0006, verified against libcint. The generic recursion is
  the reference implementation and the first version; a build-time
  generator (ADR-0002) unrolls it per angular momentum combination only
  where profiling shows the gain.
- **Boys function.** A tabulated Taylor expansion for small arguments and
  the asymptotic form for large ones, with downward recursion for lower
  orders, validated against an arbitrary-precision reference to a relative
  accuracy of 1e-14 over the orders and arguments the kernels use.
- **Screening.** Schwarz bounds for three- and four-centre integrals;
  distance-based bounds for lattice images (ADR-0007).
- **Rys quadrature is not implemented.** It may be added later for
  four-centre integrals only, behind the same kernel interface, if
  profiling of the exact-integral path demands it.
- **libcint stays the oracle.** Every native kernel with a libcint
  counterpart is compared with it at `INTEGRAL_TOL` before it replaces the
  adapter for that integral type. The pair Fourier transform, which libcint
  does not provide, is validated against numerical quadrature and by its
  value at `G = 0`, which must equal the overlap integral of the pair.

## Consequences

- One set of intermediates serves every integral type the library needs,
  and the Boys function is the only special function in the compiled
  layer. Three-centre integrals in the Hermite form are among the most
  efficient evaluations for density fitting (Ahlrichs 2006), and the Fourier
  transforms and range-separated kernels of the periodic milestones cost no
  new machinery.
- Four-centre integrals at high angular momentum are slower than with Rys
  quadrature or Head-Gordon–Pople by a small factor. This is accepted
  because the four-centre path is secondary.
- Kernel order inside M2: Boys function, one-electron integrals, two- and
  three-centre Coulomb integrals with range separation, the pair Fourier
  transform, four-centre integrals last.
- The index letters `t`, `u`, `v` of the literature for Hermite orders
  collide with the atomic-orbital letters `u`, `v` of CONVENTIONS.md §2.4;
  kernels therefore write Hermite orders as `tx`, `ty`, `tz`.
- Revisit if AO-direct exact exchange for large molecules becomes a goal,
  or if profiling shows four-centre integrals dominating a target workload;
  the response is to add Rys quadrature for four-centre integrals only.
