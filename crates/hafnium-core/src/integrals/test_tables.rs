//! Shell tables shared by the unit tests of the integral kernels.

use std::f64::consts::PI;

use super::ShellTable;

/// One normalized s primitive of exponent `a` at `center`.
pub(crate) fn s_shell(a: f64, center: [f64; 3]) -> ShellTable {
    let norm = (2.0 * a / PI).powf(0.75) * (4.0 * PI).sqrt();
    ShellTable::new(
        vec![0],
        vec![center],
        vec![0, 1],
        vec![a],
        vec![norm],
        vec![1],
    )
    .expect("valid")
}

/// An s and a p shell of two primitives each on two centres, with
/// arbitrary coefficients, translated as a whole by `shift`.
pub(crate) fn sp_table_at(shift: [f64; 3]) -> ShellTable {
    let centers = [[0.0; 3], [1.2, -0.3, 0.4]];
    let moved =
        |center: [f64; 3]| -> [f64; 3] { std::array::from_fn(|axis| center[axis] + shift[axis]) };
    ShellTable::new(
        vec![0, 1, 0, 1],
        vec![
            moved(centers[0]),
            moved(centers[0]),
            moved(centers[1]),
            moved(centers[1]),
        ],
        vec![0, 2, 4, 6, 8],
        vec![1.3, 0.4, 0.9, 0.3, 1.1, 0.35, 0.8, 0.25],
        vec![0.7, 0.5, 0.6, 0.4, 0.8, 0.3, 0.5, 0.5],
        vec![1; 4],
    )
    .expect("valid")
}

/// [`sp_table_at`] at the origin.
pub(crate) fn sp_table() -> ShellTable {
    sp_table_at([0.0; 3])
}

/// A generally contracted table: an s shell with three contractions of
/// three primitives and a p shell with two contractions of two primitives
/// on another centre, with arbitrary coefficients.
pub(crate) fn general_table() -> ShellTable {
    ShellTable::new(
        vec![0, 1],
        vec![[0.0; 3], [0.9, -0.4, 0.3]],
        vec![0, 3, 5],
        vec![2.1, 0.7, 0.25, 1.1, 0.35],
        vec![
            0.6, 0.3, 0.1, // s, contraction 0
            -0.2, 0.5, 0.4, // s, contraction 1
            0.1, -0.3, 0.8, // s, contraction 2
            0.7, 0.4, // p, contraction 0
            -0.3, 0.9, // p, contraction 1
        ],
        vec![3, 2],
    )
    .expect("valid")
}

/// [`general_table`] with every contraction as its own shell, in the same
/// function order.
pub(crate) fn split_table() -> ShellTable {
    ShellTable::new(
        vec![0, 0, 0, 1, 1],
        vec![
            [0.0; 3],
            [0.0; 3],
            [0.0; 3],
            [0.9, -0.4, 0.3],
            [0.9, -0.4, 0.3],
        ],
        vec![0, 3, 6, 9, 11, 13],
        vec![
            2.1, 0.7, 0.25, 2.1, 0.7, 0.25, 2.1, 0.7, 0.25, 1.1, 0.35, 1.1, 0.35,
        ],
        vec![
            0.6, 0.3, 0.1, -0.2, 0.5, 0.4, 0.1, -0.3, 0.8, 0.7, 0.4, -0.3, 0.9,
        ],
        vec![1; 5],
    )
    .expect("valid")
}
