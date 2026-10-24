//! Basis functions on the points of a DFT grid (PLAN.md §3.4).
//!
//! The grids themselves, the radial and angular schemes and the partition
//! of space between the atoms, are built in Python; this module evaluates
//! the contracted spherical Gaussians of a [`ShellTable`] and their
//! derivatives at a batch of points.

mod basis;

pub use basis::{basis_values, n_components};

use crate::integrals::ShellTable;
