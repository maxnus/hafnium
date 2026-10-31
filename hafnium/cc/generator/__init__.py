"""The equation generator of the coupled-cluster methods (ADR-0032).

:mod:`hafnium.cc.generator.wick` contracts normal-ordered operator strings
over the Fermi vacuum and brings the terms to a canonical form,
:mod:`hafnium.cc.generator.equations` forms the expectation values of a
projection with the Hamiltonian and cluster operators and states the
CCSD equations, the triples of (T) and the Λ equations with the density
blocks in those terms, and :mod:`hafnium.cc.generator.emit` groups the
terms by the permutations of their free indices and writes them as
programs of pairwise :func:`~hafnium.structured.contract` calls over
block tensors. ``python -m hafnium.cc.generator`` rewrites
:mod:`hafnium.cc._ccsd_equations`, :mod:`hafnium.cc._ccsd_pt_equations`
and :mod:`hafnium.cc._ccsd_lambda_equations`, which are committed and
compared with a fresh generation by the tests.
"""
