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

/// A d shell of two primitives with arbitrary coefficients.
pub(crate) fn d_shell() -> ShellTable {
    ShellTable::new(
        vec![2],
        vec![[0.2, -0.1, 0.3]],
        vec![0, 2],
        vec![0.9, 0.4],
        vec![0.6, 0.5],
        vec![1],
    )
    .expect("valid")
}

/// The functions of shell `shell` of `table` as a row range.
pub(crate) fn rows_of(table: &ShellTable, shell: usize) -> std::ops::Range<usize> {
    let start = table.function_offsets()[shell];
    start..start + table.shell(shell).n_functions()
}

/// Assert that two arrays agree to `tolerance` element by element.
pub(crate) fn assert_close(left: &[f64], right: &[f64], tolerance: f64) {
    assert_eq!(left.len(), right.len());
    for (l, r) in left.iter().zip(right) {
        assert!((l - r).abs() < tolerance, "{l} vs {r}");
    }
}

/// `table` with the centre of shell `shell` moved by `delta` along `axis`.
pub(crate) fn moved(table: &ShellTable, shell: usize, axis: usize, delta: f64) -> ShellTable {
    let mut moved = table.clone();
    moved.centers[shell][axis] += delta;
    moved
}

/// The step of the central differences the derivative tests take.
pub(crate) const STEP: f64 = 1e-5;

/// The central difference of `evaluate` with respect to the centre of
/// `shell` of `table` along `axis`.
pub(crate) fn central_difference(
    table: &ShellTable,
    shell: usize,
    axis: usize,
    evaluate: impl Fn(&ShellTable) -> Vec<f64>,
) -> Vec<f64> {
    let plus = evaluate(&moved(table, shell, axis, STEP));
    let minus = evaluate(&moved(table, shell, axis, -STEP));
    plus.iter()
        .zip(&minus)
        .map(|(p, m)| (p - m) / (2.0 * STEP))
        .collect()
}

/// The central difference of `evaluate` with respect to `vector[axis]`.
pub(crate) fn vector_difference(
    vector: [f64; 3],
    axis: usize,
    evaluate: impl Fn([f64; 3]) -> Vec<f64>,
) -> Vec<f64> {
    let mut plus = vector;
    plus[axis] += STEP;
    let mut minus = vector;
    minus[axis] -= STEP;
    evaluate(plus)
        .iter()
        .zip(&evaluate(minus))
        .map(|(p, m)| (p - m) / (2.0 * STEP))
        .collect()
}

/// Assert that the rows `rows` of every image of `derivative`, laid out
/// `[image][n_slots][n_rows][row_size]` and read at `slot`, agree with the
/// same rows of `difference`, laid out `[image][n_rows][row_size]`.
pub(crate) fn assert_rows_match(
    derivative: &[f64],
    (n_slots, slot): (usize, usize),
    (n_rows, row_size): (usize, usize),
    rows: std::ops::Range<usize>,
    difference: &[f64],
) {
    let image_size = n_rows * row_size;
    let n_images = difference.len() / image_size;
    assert_eq!(derivative.len(), n_images * n_slots * image_size);
    for image in 0..n_images {
        for row in rows.clone() {
            for column in 0..row_size {
                let analytic =
                    derivative[((image * n_slots + slot) * n_rows + row) * row_size + column];
                let numeric = difference[(image * n_rows + row) * row_size + column];
                assert!(
                    (analytic - numeric).abs() < 1e-8,
                    "slot {slot} row {row} column {column}: {analytic} vs {numeric}"
                );
            }
        }
    }
}
