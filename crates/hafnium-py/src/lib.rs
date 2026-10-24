//! Python bindings for `hafnium-core`, exposed as `hafnium._kernels`.
//!
//! This crate is deliberately thin: it converts Python arrays to slices,
//! calls `hafnium-core`, and maps errors to Python exceptions. No numerics
//! live here.

use hafnium_core::{integrals, periodic};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArray5, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
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

/// Contracted shells in structure-of-arrays form, one entry per shell with
/// all its contractions (see `hafnium.integrals`).
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
        n_contractions: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Self> {
        let inner = integrals::ShellTable::new(
            to_indices(angular_momenta.as_slice()?, "angular momenta")?,
            to_vectors(&centers, "centres")?,
            to_indices(primitive_offsets.as_slice()?, "primitive offsets")?,
            exponents.as_slice()?.to_vec(),
            coefficients.as_slice()?.to_vec(),
            to_indices(n_contractions.as_slice()?, "contraction counts")?,
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
/// the ket, `[image, bra, ket]`; `omega` selects the `erf(ωr)/r` kernel,
/// or `erfc(ωr)/r` with `short_range`, and terms bounded below
/// `threshold` are skipped.
#[pyfunction]
#[pyo3(signature = (bra, ket, translations, charges, positions, omega=None, short_range=false, threshold=0.0))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the binding mirrors the kernel's arguments"
)]
fn nuclear_attraction<'py>(
    py: Python<'py>,
    bra: &PyShellTable,
    ket: &PyShellTable,
    translations: PyReadonlyArray2<'py, f64>,
    charges: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray2<'py, f64>,
    omega: Option<f64>,
    short_range: bool,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let translations = to_vectors(&translations, "translations")?;
    let positions = to_vectors(&positions, "positions")?;
    let kernel = to_kernel(omega, short_range)?;
    let values = integrals::nuclear_attraction(
        &bra.inner,
        &ket.inner,
        &translations,
        charges.as_slice()?,
        &positions,
        kernel,
        threshold,
    )
    .map_err(|error| value_error(&error))?;
    to_image_matrices(py, values, translations.len(), bra, ket)
}

/// Two-centre Coulomb integrals for each translation of the ket,
/// `[image, bra, ket]`; `omega` selects the `erf(ωr)/r` kernel, or
/// `erfc(ωr)/r` with `short_range`, and terms bounded below `threshold`
/// are skipped.
#[pyfunction]
#[pyo3(signature = (bra, ket, translations, omega=None, short_range=false, threshold=0.0))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn coulomb_two_center<'py>(
    py: Python<'py>,
    bra: &PyShellTable,
    ket: &PyShellTable,
    translations: PyReadonlyArray2<'py, f64>,
    omega: Option<f64>,
    short_range: bool,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let translations = to_vectors(&translations, "translations")?;
    let kernel = to_kernel(omega, short_range)?;
    let values =
        integrals::coulomb_two_center(&bra.inner, &ket.inner, &translations, kernel, threshold)
            .map_err(|error| value_error(&error))?;
    to_image_matrices(py, values, translations.len(), bra, ket)
}

/// Three-centre Coulomb integrals `(P | μ ν)` for each pair of translations
/// (auxiliary, ket), `[image, aux, bra, ket]`.
#[pyfunction]
#[pyo3(signature = (auxiliary, bra, ket, translations, omega=None, short_range=false, threshold=0.0))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the binding mirrors the kernel's arguments"
)]
fn coulomb_three_center<'py>(
    py: Python<'py>,
    auxiliary: &PyShellTable,
    bra: &PyShellTable,
    ket: &PyShellTable,
    translations: PyReadonlyArray3<'py, f64>,
    omega: Option<f64>,
    short_range: bool,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray4<f64>>> {
    let translations = to_vector_sets::<2>(&translations)?;
    let kernel = to_kernel(omega, short_range)?;
    let values = integrals::coulomb_three_center(
        &auxiliary.inner,
        &bra.inner,
        &ket.inner,
        &translations,
        kernel,
        threshold,
    )
    .map_err(|error| value_error(&error))?;
    let shape = [
        translations.len(),
        auxiliary.inner.n_functions(),
        bra.inner.n_functions(),
        ket.inner.n_functions(),
    ];
    values.into_pyarray(py).reshape(shape)
}

/// Three-centre Coulomb integrals summed over the image pairs `(i, j)` of
/// the orbital functions with phases, complex `[kpt, kpt, aux, bra, ket]`:
/// `Σ_{ij} bra_phases[k1, i] ket_phases[k2, j] (P | μ(T_i) ν(T_j))` with
/// the auxiliary function in the home cell, over the pairs whose
/// difference `T_j − T_i` is one of the `steps`, found through the
/// integer lattice `coordinates` of the images and of the steps.
#[pyfunction]
#[pyo3(signature = (auxiliary, bra, ket, images, image_coordinates, steps, step_coordinates, image_classes, kpoints, omega=None, short_range=false, threshold=0.0))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the binding mirrors the kernel's arguments"
)]
fn coulomb_three_center_bloch<'py>(
    py: Python<'py>,
    auxiliary: &PyShellTable,
    bra: &PyShellTable,
    ket: &PyShellTable,
    images: PyReadonlyArray2<'py, f64>,
    image_coordinates: PyReadonlyArray2<'py, i64>,
    steps: PyReadonlyArray2<'py, f64>,
    step_coordinates: PyReadonlyArray2<'py, i64>,
    image_classes: PyReadonlyArray1<'py, i64>,
    kpoints: PyReadonlyArray2<'py, f64>,
    omega: Option<f64>,
    short_range: bool,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray5<Complex64>>> {
    let images = to_vectors(&images, "translations")?;
    let steps = to_vectors(&steps, "translations")?;
    let image_coordinates = to_coordinates(&image_coordinates)?;
    let step_coordinates = to_coordinates(&step_coordinates)?;
    let image_classes = to_indices(image_classes.as_slice()?, "image classes")?;
    let kpoints = to_vectors(&kpoints, "k-points")?;
    let n_k = kpoints.len();
    let kernel = to_kernel(omega, short_range)?;
    let sum = integrals::BlochSum {
        images: &images,
        image_coordinates: &image_coordinates,
        steps: &steps,
        step_coordinates: &step_coordinates,
        image_classes: &image_classes,
        kpoints: &kpoints,
    };
    let values = integrals::coulomb_three_center_bloch(
        &auxiliary.inner,
        &bra.inner,
        &ket.inner,
        sum,
        kernel,
        threshold,
    )
    .map_err(|error| value_error(&error))?;
    let shape = [
        n_k,
        n_k,
        auxiliary.inner.n_functions(),
        bra.inner.n_functions(),
        ket.inner.n_functions(),
    ];
    values.into_pyarray(py).reshape(shape)
}

/// Four-centre Coulomb integrals `(uv|wx)` for each triple of translations
/// (second bra, first ket, second ket), `[image, ao, ao, ao, ao]`; shell
/// quartets with a Schwarz bound below `threshold` are left at zero.
#[pyfunction]
#[pyo3(signature = (bra, ket, translations, omega=None, short_range=false, threshold=0.0))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn coulomb_four_center<'py>(
    py: Python<'py>,
    bra: (PyRef<'py, PyShellTable>, PyRef<'py, PyShellTable>),
    ket: (PyRef<'py, PyShellTable>, PyRef<'py, PyShellTable>),
    translations: PyReadonlyArray3<'py, f64>,
    omega: Option<f64>,
    short_range: bool,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray5<f64>>> {
    let translations = to_vector_sets::<3>(&translations)?;
    let bra = [&bra.0.inner, &bra.1.inner];
    let ket = [&ket.0.inner, &ket.1.inner];
    let kernel = to_kernel(omega, short_range)?;
    let values = integrals::coulomb_four_center(bra, ket, &translations, kernel, threshold)
        .map_err(|error| value_error(&error))?;
    let shape = [
        translations.len(),
        bra[0].n_functions(),
        bra[1].n_functions(),
        ket[0].n_functions(),
        ket[1].n_functions(),
    ];
    values.into_pyarray(py).reshape(shape)
}

/// Schwarz bounds `max sqrt((ab|ab))` of every shell pair for each
/// translation of the ket, `[image, bra_shell, ket_shell]`.
#[pyfunction]
#[pyo3(signature = (bra, ket, translations, omega=None, short_range=false))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn schwarz_bounds<'py>(
    py: Python<'py>,
    bra: &PyShellTable,
    ket: &PyShellTable,
    translations: PyReadonlyArray2<'py, f64>,
    omega: Option<f64>,
    short_range: bool,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let translations = to_vectors(&translations, "translations")?;
    let kernel = to_kernel(omega, short_range)?;
    let values = integrals::schwarz_bounds(&bra.inner, &ket.inner, &translations, kernel)
        .map_err(|error| value_error(&error))?;
    let shape = [
        translations.len(),
        bra.inner.n_shells(),
        ket.inner.n_shells(),
    ];
    values.into_pyarray(py).reshape(shape)
}

/// Fourier transforms `∫ χ(r) exp(−i G·r) dr` of every function for each
/// vector in `g_vectors` (inverse Bohr), complex `[G, ao]`.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn fourier_transform<'py>(
    py: Python<'py>,
    table: &PyShellTable,
    g_vectors: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let g_vectors = to_vectors(&g_vectors, "G vectors")?;
    let values = integrals::fourier_transform(&table.inner, &g_vectors)
        .map_err(|error| value_error(&error))?;
    values
        .into_pyarray(py)
        .reshape([g_vectors.len(), table.inner.n_functions()])
}

/// Pair transforms `∫ φ_u(r) φ_v(r − T) exp(−i G·r) dr` for each
/// translation `T` of the ket, complex `[image, G, ao, ao]`; pairs bounded
/// below `threshold` are skipped.
#[pyfunction]
#[pyo3(signature = (bra, ket, translations, g_vectors, threshold=0.0))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn pair_fourier_transform<'py>(
    py: Python<'py>,
    bra: &PyShellTable,
    ket: &PyShellTable,
    translations: PyReadonlyArray2<'py, f64>,
    g_vectors: PyReadonlyArray2<'py, f64>,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray4<Complex64>>> {
    let translations = to_vectors(&translations, "translations")?;
    let g_vectors = to_vectors(&g_vectors, "G vectors")?;
    let values = integrals::pair_fourier_transform(
        &bra.inner,
        &ket.inner,
        &translations,
        &g_vectors,
        threshold,
    )
    .map_err(|error| value_error(&error))?;
    let shape = [
        translations.len(),
        g_vectors.len(),
        bra.inner.n_functions(),
        ket.inner.n_functions(),
    ];
    values.into_pyarray(py).reshape(shape)
}

/// Ewald energy per cell of point `charges` at `positions` on the lattice
/// spanned by `vectors`, summed over the real-space `translations` and the
/// non-zero reciprocal `g_vectors`, with the range-separation parameter `omega`.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn ewald_energy(
    vectors: PyReadonlyArray2<'_, f64>,
    charges: PyReadonlyArray1<'_, f64>,
    positions: PyReadonlyArray2<'_, f64>,
    translations: PyReadonlyArray2<'_, f64>,
    g_vectors: PyReadonlyArray2<'_, f64>,
    omega: f64,
) -> PyResult<f64> {
    let charges = charges.as_slice()?;
    if !charges.iter().all(|z| z.is_finite()) {
        return Err(PyValueError::new_err("charges are finite"));
    }
    periodic::ewald_energy(
        &to_vectors(&vectors, "lattice vectors")?,
        charges,
        &to_vectors(&positions, "positions")?,
        &to_vectors(&translations, "translations")?,
        &to_vectors(&g_vectors, "G vectors")?,
        omega,
    )
    .map_err(|error| value_error(&error))
}

/// Set the size of the kernel thread pool once per process.
#[pyfunction]
fn configure_threads(n_threads: usize) -> PyResult<()> {
    integrals::configure_threads(n_threads).map_err(|error| value_error(&error))
}

fn to_kernel(omega: Option<f64>, short_range: bool) -> PyResult<integrals::CoulombKernel> {
    integrals::CoulombKernel::new(omega, short_range).map_err(|error| value_error(&error))
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
    let values = array.as_slice()?;
    if !values.iter().all(|x| x.is_finite()) {
        return Err(PyValueError::new_err(format!("{name} are finite")));
    }
    Ok(values
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect())
}

/// Rows of an `[n, 3]` integer array as lattice coordinates.
fn to_coordinates(array: &PyReadonlyArray2<'_, i64>) -> PyResult<Vec<[i64; 3]>> {
    if array.shape()[1] != 3 {
        return Err(PyValueError::new_err("coordinates have shape [n, 3]"));
    }
    Ok(array
        .as_slice()?
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect())
}

/// Rows of an `[n, N, 3]` array as sets of `N` vectors.
fn to_vector_sets<const N: usize>(
    array: &PyReadonlyArray3<'_, f64>,
) -> PyResult<Vec<[[f64; 3]; N]>> {
    if array.shape()[1..] != [N, 3] {
        return Err(PyValueError::new_err(format!(
            "translations have shape [n, {N}, 3]"
        )));
    }
    let values = array.as_slice()?;
    if !values.iter().all(|x| x.is_finite()) {
        return Err(PyValueError::new_err("translations are finite"));
    }
    Ok(values
        .chunks_exact(3 * N)
        .map(|row| std::array::from_fn(|i| [row[3 * i], row[3 * i + 1], row[3 * i + 2]]))
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
    module.add_function(wrap_pyfunction!(coulomb_two_center, module)?)?;
    module.add_function(wrap_pyfunction!(coulomb_three_center, module)?)?;
    module.add_function(wrap_pyfunction!(coulomb_three_center_bloch, module)?)?;
    module.add_function(wrap_pyfunction!(coulomb_four_center, module)?)?;
    module.add_function(wrap_pyfunction!(schwarz_bounds, module)?)?;
    module.add_function(wrap_pyfunction!(fourier_transform, module)?)?;
    module.add_function(wrap_pyfunction!(pair_fourier_transform, module)?)?;
    module.add_function(wrap_pyfunction!(ewald_energy, module)?)?;
    module.add_function(wrap_pyfunction!(configure_threads, module)?)?;
    Ok(())
}
