//! Python bindings for `hafnium-core`, exposed as `hafnium._kernels`.
//!
//! This crate is deliberately thin: it converts Python arrays to slices,
//! calls `hafnium-core`, and maps errors to Python exceptions. No numerics
//! live here.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Return the version of the compiled kernel crate.
#[pyfunction]
fn kernel_version() -> &'static str {
    hafnium_core::VERSION
}

/// Dot product of two contiguous one-dimensional float64 arrays.
///
/// Bootstrap smoke test of the array boundary. Raises `ValueError` on a
/// length mismatch and `TypeError` on non-contiguous input.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn dot(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let a = a.as_slice()?;
    let b = b.as_slice()?;
    hafnium_core::dot(a, b).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pymodule]
fn _kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(kernel_version, module)?)?;
    module.add_function(wrap_pyfunction!(dot, module)?)?;
    Ok(())
}
