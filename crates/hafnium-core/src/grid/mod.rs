//! Basis functions and partition weights on the points of a DFT grid
//! (PLAN.md §3.4).
//!
//! The grids themselves, the radial and angular schemes and the assembly
//! of the atomic grids, are built in Python; this module evaluates the
//! contracted spherical Gaussians of a [`ShellTable`] and their
//! derivatives at a batch of points, their Bloch sums over lattice images
//! for a cell, and the partition weights of an atom among many centres.

mod basis;
mod bloch;
mod partition;

pub use basis::{basis_values, n_components};
pub use bloch::bloch_basis_values;
pub use partition::{Partition, partition_weights};

use crate::integrals::ShellTable;
