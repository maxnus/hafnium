"""The equation generator of the coupled-cluster methods (ADR-0032).

:mod:`hafnium.cc.generator.wick` contracts normal-ordered operator strings
over the Fermi vacuum and brings the terms to a canonical form,
:mod:`hafnium.cc.generator.ccsd` states the CCSD equations in those terms,
and :mod:`hafnium.cc.generator.emit` groups the terms by the permutations
of their free indices and writes them as a program of pairwise
:func:`~hafnium.structured.contract` calls over block tensors. ``python
-m hafnium.cc.generator`` rewrites :mod:`hafnium.cc._ccsd_equations`,
which is committed and compared with a fresh generation by the tests.
"""
