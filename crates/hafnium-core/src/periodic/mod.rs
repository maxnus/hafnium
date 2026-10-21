//! Lattice sums of periodic systems (ADR-0007).
//!
//! The image lists and cutoffs are chosen in Python; this module runs the
//! loops over them.

mod ewald;

pub use ewald::ewald_energy;
