//! Python bindings for `hafnium-core`, exposed as `hafnium._kernels`.
//!
//! This crate is deliberately thin: it converts Python arrays to slices,
//! calls `hafnium-core`, and maps errors to Python exceptions. No numerics
//! live here.

use hafnium_core::integrals;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
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
    hafnium_core::dot(a, b).map_err(|error| value_error(&error))
}

/// The Boys function `F_n(t)` for `n = 0, ..., n_max`.
#[pyfunction]
fn boys_function(py: Python<'_>, n_max: usize, t: f64) -> PyResult<Bound<'_, PyArray1<f64>>> {
    if t.is_nan() || t < 0.0 {
        return Err(PyValueError::new_err(
            "the Boys function argument is non-negative",
        ));
    }
    Ok(hafnium_core::special::boys_values(n_max, t).into_pyarray(py))
}

/// Transformation from Cartesian monomials to real solid harmonics for
/// angular momentum `l`, shape `[2l + 1, (l + 1)(l + 2) / 2]`.
#[pyfunction]
fn spherical_transformation(py: Python<'_>, l: usize) -> PyResult<Bound<'_, PyArray2<f64>>> {
    if l > integrals::MAX_ANGULAR_MOMENTUM {
        return Err(PyValueError::new_err(
            "angular momentum exceeds the supported maximum",
        ));
    }
    let matrix = integrals::spherical_transformation(l);
    matrix
        .into_pyarray(py)
        .reshape([2 * l + 1, integrals::n_cartesian(l)])
}

/// Contracted shells in structure-of-arrays form (see `hafnium.integrals`).
#[pyclass(frozen, name = "ShellTable")]
struct PyShellTable {
    inner: integrals::ShellTable,
}

#[pymethods]
impl PyShellTable {
    #[new]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts #[pyfunction] arguments by value"
    )]
    fn new(
        angular_momenta: PyReadonlyArray1<'_, i64>,
        centers: PyReadonlyArray2<'_, f64>,
        primitive_offsets: PyReadonlyArray1<'_, i64>,
        exponents: PyReadonlyArray1<'_, f64>,
        coefficients: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let inner = integrals::ShellTable::new(
            to_indices(angular_momenta.as_slice()?, "angular momenta")?,
            to_vectors(&centers, "centres")?,
            to_indices(primitive_offsets.as_slice()?, "primitive offsets")?,
            exponents.as_slice()?.to_vec(),
            coefficients.as_slice()?.to_vec(),
        )
        .map_err(|error| value_error(&error))?;
        Ok(Self { inner })
    }

    /// Number of shells.
    #[getter]
    fn n_shells(&self) -> usize {
        self.inner.n_shells()
    }

    /// Number of spherical basis functions.
    #[getter]
    fn n_functions(&self) -> usize {
        self.inner.n_functions()
    }
}

/// Overlap integrals for each translation of the ket, `[image, bra, ket]`.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn overlap<'py>(
    py: Python<'py>,
    bra: &PyShellTable,
    ket: &PyShellTable,
    translations: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let translations = to_vectors(&translations, "translations")?;
    let values = integrals::overlap(&bra.inner, &ket.inner, &translations);
    to_image_matrices(py, values, translations.len(), bra, ket)
}

/// Kinetic energy integrals for each translation of the ket, `[image, bra, ket]`.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn kinetic<'py>(
    py: Python<'py>,
    bra: &PyShellTable,
    ket: &PyShellTable,
    translations: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let translations = to_vectors(&translations, "translations")?;
    let values = integrals::kinetic(&bra.inner, &ket.inner, &translations);
    to_image_matrices(py, values, translations.len(), bra, ket)
}

/// Nuclear attraction integrals of point charges for each translation of
/// the ket, `[image, bra, ket]`; `omega` selects the `erf(ωr)/r` kernel.
#[pyfunction]
#[pyo3(signature = (bra, ket, translations, charges, positions, omega=None))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn nuclear_attraction<'py>(
    py: Python<'py>,
    bra: &PyShellTable,
    ket: &PyShellTable,
    translations: PyReadonlyArray2<'py, f64>,
    charges: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray2<'py, f64>,
    omega: Option<f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let translations = to_vectors(&translations, "translations")?;
    let positions = to_vectors(&positions, "positions")?;
    let values = integrals::nuclear_attraction(
        &bra.inner,
        &ket.inner,
        &translations,
        charges.as_slice()?,
        &positions,
        omega,
    )
    .map_err(|error| value_error(&error))?;
    to_image_matrices(py, values, translations.len(), bra, ket)
}

fn value_error(error: &hafnium_core::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Non-negative integers from a signed array.
fn to_indices(values: &[i64], name: &str) -> PyResult<Vec<usize>> {
    values
        .iter()
        .map(|&value| usize::try_from(value))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| PyValueError::new_err(format!("{name} are non-negative integers")))
}

/// Rows of an `[n, 3]` array as vectors.
fn to_vectors(array: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<Vec<[f64; 3]>> {
    if array.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!("{name} have shape [n, 3]")));
    }
    Ok(array
        .as_slice()?
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect())
}

fn to_image_matrices<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    n_images: usize,
    bra: &PyShellTable,
    ket: &PyShellTable,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let shape = [n_images, bra.inner.n_functions(), ket.inner.n_functions()];
    values.into_pyarray(py).reshape(shape)
}

#[pymodule]
fn _kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyShellTable>()?;
    module.add_function(wrap_pyfunction!(kernel_version, module)?)?;
    module.add_function(wrap_pyfunction!(dot, module)?)?;
    module.add_function(wrap_pyfunction!(boys_function, module)?)?;
    module.add_function(wrap_pyfunction!(spherical_transformation, module)?)?;
    module.add_function(wrap_pyfunction!(overlap, module)?)?;
    module.add_function(wrap_pyfunction!(kinetic, module)?)?;
    module.add_function(wrap_pyfunction!(nuclear_attraction, module)?)?;
    Ok(())
}
