//! Python bindings for `hafnium-core`, exposed as `hafnium._kernels`.
//!
//! This crate is deliberately thin: it converts Python arrays to slices,
//! calls `hafnium-core`, and maps errors to Python exceptions. No numerics
//! live here.

use hafnium_core::{grid, integrals, periodic, xc};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArray5, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

/// Bloch-summed three-centre Coulomb integrals as complex
/// `[kpt, kpt, aux, bra, ket]`, `Σ_ij exp(−i k1·T_i) exp(i k2·T_j)
/// (P | u(T_i) v(T_j))` over the pairs of `images` whose difference is one
/// of the `steps`, found through the integer lattice coordinates of both,
/// the auxiliary function in the home cell; `exp(i (k2 − k1)·T_i)` is
/// taken from the class of every image and `exp(i k2·T_s)` from the class
/// of every step (ADR-0025, ADR-0028). With `coulomb_only` only the blocks
/// of zero momentum transfer are formed, `[kpt, aux, bra, ket]`.
#[pyfunction]
#[pyo3(signature = (auxiliary, bra, ket, images, image_coordinates, steps, step_coordinates, image_classes, step_classes, kpoints, omega=None, short_range=false, threshold=0.0, coulomb_only=false))]
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
    step_classes: PyReadonlyArray1<'py, i64>,
    kpoints: PyReadonlyArray2<'py, f64>,
    omega: Option<f64>,
    short_range: bool,
    threshold: f64,
    coulomb_only: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let images = to_vectors(&images, "translations")?;
    let steps = to_vectors(&steps, "translations")?;
    let image_coordinates = to_coordinates(&image_coordinates)?;
    let step_coordinates = to_coordinates(&step_coordinates)?;
    let image_classes = to_indices(image_classes.as_slice()?, "image classes")?;
    let step_classes = to_indices(step_classes.as_slice()?, "step classes")?;
    let kpoints = to_vectors(&kpoints, "k-points")?;
    let n_k = kpoints.len();
    let kernel = to_kernel(omega, short_range)?;
    let sum = integrals::BlochSum {
        images: &images,
        image_coordinates: &image_coordinates,
        steps: &steps,
        step_coordinates: &step_coordinates,
        image_classes: &image_classes,
        step_classes: &step_classes,
        kpoints: &kpoints,
    };
    let transfers = if coulomb_only {
        integrals::Transfers::Zero
    } else {
        integrals::Transfers::All
    };
    let values = integrals::coulomb_three_center_bloch(
        &auxiliary.inner,
        &bra.inner,
        &ket.inner,
        sum,
        kernel,
        threshold,
        transfers,
    )
    .map_err(|error| value_error(&error))?;
    let functions = [
        auxiliary.inner.n_functions(),
        bra.inner.n_functions(),
        ket.inner.n_functions(),
    ];
    let array = values.into_pyarray(py);
    if coulomb_only {
        let shape = [n_k, functions[0], functions[1], functions[2]];
        Ok(array.reshape(shape)?.into_any())
    } else {
        let shape = [n_k, n_k, functions[0], functions[1], functions[2]];
        Ok(array.reshape(shape)?.into_any())
    }
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

/// Values of every function of `table` at `points` (`[g, 3]`, Bohr) with
/// derivatives up to `order` (at most two), `[g, component, ao]`; the
/// components are the value, the derivatives `x, y, z` and the second
/// derivatives `xx, xy, xz, yy, yz, zz`. A shell's functions are zero at
/// points farther than its entry of `radii` from its centre.
#[pyfunction]
#[pyo3(signature = (table, points, radii, order=0))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn basis_values<'py>(
    py: Python<'py>,
    table: &PyShellTable,
    points: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    order: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let points = to_vectors(&points, "points")?;
    let values = grid::basis_values(&table.inner, &points, radii.as_slice()?, order)
        .map_err(|error| value_error(&error))?;
    let shape = [
        points.len(),
        grid::n_components(order),
        table.inner.n_functions(),
    ];
    values.into_pyarray(py).reshape(shape)
}

/// Bloch sums `Σ_T exp(i k·T) φ(r − T)` of every function of `table` at
/// `points` (`[g, 3]`, Bohr) with derivatives up to `order`, as complex
/// `[kpt, g, component, ao]`, over the `translations` (`[image, 3]`, Bohr)
/// with the `phases` `exp(i k·T)` given as `[kpt, image]`; the cutoff
/// `radii` apply to every image of a shell as in `basis_values`.
#[pyfunction]
#[pyo3(signature = (table, points, radii, translations, phases, order=0))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn bloch_basis_values<'py>(
    py: Python<'py>,
    table: &PyShellTable,
    points: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    translations: PyReadonlyArray2<'py, f64>,
    phases: PyReadonlyArray2<'py, Complex64>,
    order: usize,
) -> PyResult<Bound<'py, PyArray4<Complex64>>> {
    let points = to_vectors(&points, "points")?;
    let translations = to_vectors(&translations, "translations")?;
    let n_kpt = phases.shape()[0];
    if phases.shape()[1] != translations.len() {
        return Err(PyValueError::new_err(
            "phases have shape [kpt, image] with one entry per translation",
        ));
    }
    let values = grid::bloch_basis_values(
        &table.inner,
        &points,
        radii.as_slice()?,
        &translations,
        phases.as_slice()?,
        order,
    )
    .map_err(|error| value_error(&error))?;
    let shape = [
        n_kpt,
        points.len(),
        grid::n_components(order),
        table.inner.n_functions(),
    ];
    values.into_pyarray(py).reshape(shape)
}

/// Partition weights `[g]` of the centre `owner` at `points` (`[g, 3]`,
/// Bohr) among `centers` (`[centre, 3]`): the normalized cell function of
/// Becke (`scheme` `"becke"`) or of Stratmann, Scuseria and Frisch
/// (`"stratmann"`), with the atomic size `adjustments` `[centre, centre]`
/// when given.
#[pyfunction]
#[pyo3(signature = (points, centers, owner, adjustments=None, scheme="becke"))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
fn partition_weights<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    owner: usize,
    adjustments: Option<PyReadonlyArray2<'py, f64>>,
    scheme: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let points = to_vectors(&points, "points")?;
    let centers = to_vectors(&centers, "centres")?;
    let scheme = match scheme {
        "becke" => grid::Partition::Becke,
        "stratmann" => grid::Partition::Stratmann,
        _ => {
            return Err(PyValueError::new_err(
                "the partition scheme is \"becke\" or \"stratmann\"",
            ));
        }
    };
    let table = match &adjustments {
        Some(array) => {
            if array.shape() != [centers.len(), centers.len()] {
                return Err(PyValueError::new_err(
                    "adjustments have shape [centre, centre]",
                ));
            }
            Some(array.as_slice()?)
        }
        None => None,
    };
    let weights = grid::partition_weights(&points, &centers, owner, table, scheme)
        .map_err(|error| value_error(&error))?;
    Ok(weights.into_pyarray(py))
}

/// Metadata of the functional with the libxc `name`: its id, canonical
/// name, rung (`"lda"`, `"gga"` or `"meta-gga"`), whether it needs the
/// Laplacian, the fraction of exact exchange, the range separation
/// `(omega, alpha, beta)` or `None`, the VV10 parameters `(b, C)` or
/// `None`, and the density threshold.
#[pyfunction]
fn functional_info<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
    let info = xc::functional_info(name).map_err(|error| value_error(&error))?;
    let dict = PyDict::new(py);
    dict.set_item("id", info.id)?;
    dict.set_item("name", info.name)?;
    dict.set_item("rung", info.rung.name())?;
    dict.set_item("needs_laplacian", info.needs_laplacian)?;
    dict.set_item("exact_exchange", info.exact_exchange)?;
    dict.set_item("range_separation", info.range_separation)?;
    dict.set_item("vv10", info.vv10)?;
    dict.set_item("density_threshold", info.density_threshold)?;
    Ok(dict)
}

/// Energy per particle `[g]` and the derivatives `vrho [2, g]`,
/// `vsigma [3, g]` and `vtau [2, g]` of `Σ_i weights[i] names[i]` at the
/// spin densities `rho [2, g]`, contracted gradients `sigma [3, g]` and
/// kinetic energy densities `tau [2, g]`; `vsigma` and `vtau` are `None`
/// below the rung that needs them, and `sigma` and `tau` may be `None`
/// then.
#[pyfunction]
#[pyo3(signature = (names, weights, rho, sigma=None, tau=None))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
#[expect(
    clippy::type_complexity,
    reason = "the binding returns the four arrays of the kernel"
)]
fn evaluate_functional<'py>(
    py: Python<'py>,
    names: Vec<String>,
    weights: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray2<'py, f64>,
    sigma: Option<PyReadonlyArray2<'py, f64>>,
    tau: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
    Option<Bound<'py, PyArray2<f64>>>,
)> {
    let weights = weights.as_slice()?;
    if weights.len() != names.len() {
        return Err(PyValueError::new_err("one weight per functional name"));
    }
    let components: Vec<(String, f64)> = names.into_iter().zip(weights.iter().copied()).collect();
    if rho.shape()[0] != 2 {
        return Err(PyValueError::new_err("densities have shape [2, g]"));
    }
    let n = rho.shape()[1];
    let sigma = to_rows(sigma.as_ref(), 3, n, "contracted gradients")?;
    let tau = to_rows(tau.as_ref(), 2, n, "kinetic energy densities")?;
    let values = xc::evaluate_functional(&components, rho.as_slice()?, sigma, tau)
        .map_err(|error| value_error(&error))?;
    let rows = |data: Vec<f64>, parts: usize| -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        if data.is_empty() {
            Ok(None)
        } else {
            Ok(Some(data.into_pyarray(py).reshape([parts, n])?))
        }
    };
    let vrho = values.vrho.into_pyarray(py).reshape([2, n])?;
    Ok((
        values.energy_per_particle.into_pyarray(py),
        vrho,
        rows(values.vsigma, 3)?,
        rows(values.vtau, 2)?,
    ))
}

/// Multi-collinear evaluation of `Σ weights[i] names[i]` for a noncollinear
/// density (ADR-0029): the energy density and the derivatives with respect
/// to `rho [g]`, `m [3, g]`, `grad_rho [3, g]`, `grad_m [3, 3, g]`,
/// `tau [g]` and `tau_m [3, g]`, in the layouts of the inputs, the
/// derivatives of the variables a rung does not read being `None`, and
/// the number of points whose adaptive rule did not converge. The sphere
/// is sampled at `directions [k, 3]` with `direction_weights [k]` summing
/// to one when given, else with rules aligned with the magnetization
/// refined to `precision` (scaled by `point_weights [g]`) up to
/// `max_level`; points with `|m| ≥ collinear_threshold · rho` take the
/// locally collinear functional.
#[pyfunction]
#[pyo3(signature = (
    names, weights, rho, m, grad_rho=None, grad_m=None, tau=None, tau_m=None,
    directions=None, direction_weights=None, precision=1e-14, max_level=6,
    collinear_threshold=None, point_weights=None
))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts #[pyfunction] arguments by value"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "one argument per density variable and rule part"
)]
#[expect(
    clippy::type_complexity,
    reason = "the binding returns the seven arrays of the kernel"
)]
fn evaluate_noncollinear<'py>(
    py: Python<'py>,
    names: Vec<String>,
    weights: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    m: PyReadonlyArray2<'py, f64>,
    grad_rho: Option<PyReadonlyArray2<'py, f64>>,
    grad_m: Option<PyReadonlyArray3<'py, f64>>,
    tau: Option<PyReadonlyArray1<'py, f64>>,
    tau_m: Option<PyReadonlyArray2<'py, f64>>,
    directions: Option<PyReadonlyArray2<'py, f64>>,
    direction_weights: Option<PyReadonlyArray1<'py, f64>>,
    precision: f64,
    max_level: usize,
    collinear_threshold: Option<f64>,
    point_weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
    Option<Bound<'py, PyArray3<f64>>>,
    Option<Bound<'py, PyArray1<f64>>>,
    Option<Bound<'py, PyArray2<f64>>>,
    usize,
)> {
    let weights = weights.as_slice()?;
    if weights.len() != names.len() {
        return Err(PyValueError::new_err("one weight per functional name"));
    }
    let components: Vec<(String, f64)> = names.into_iter().zip(weights.iter().copied()).collect();
    let n = rho.shape()[0];
    if m.shape() != [3, n] {
        return Err(PyValueError::new_err(
            "the magnetization has shape [3, g] over the points of the density",
        ));
    }
    let grad_m = match grad_m.as_ref() {
        None => &[][..],
        Some(array) => {
            if array.shape() != [3, 3, n] {
                return Err(PyValueError::new_err(
                    "magnetization gradients have shape [3, 3, g] over the points of the density",
                ));
            }
            array.as_slice()?
        }
    };
    let tau = match tau.as_ref() {
        None => &[][..],
        Some(array) => {
            if array.shape() != [n] {
                return Err(PyValueError::new_err(
                    "kinetic energy densities have shape [g] over the points of the density",
                ));
            }
            array.as_slice()?
        }
    };
    let sampling = sampling_from(
        directions.as_ref(),
        direction_weights.as_ref(),
        precision,
        max_level,
    )?;
    let point_weights = match point_weights.as_ref() {
        None => None,
        Some(array) => {
            if array.shape() != [n] {
                return Err(PyValueError::new_err(
                    "point weights have shape [g] over the points of the density",
                ));
            }
            Some(array.as_slice()?)
        }
    };
    let input = xc::NoncollinearInput {
        rho: rho.as_slice()?,
        m: m.as_slice()?,
        grad_rho: to_rows(grad_rho.as_ref(), 3, n, "density gradients")?,
        grad_m,
        tau,
        tau_m: to_rows(tau_m.as_ref(), 3, n, "spin kinetic energy densities")?,
    };
    let values = xc::evaluate_noncollinear(
        &components,
        &input,
        sampling,
        collinear_threshold,
        point_weights,
    )
    .map_err(|error| value_error(&error))?;
    let rows = |data: Vec<f64>, parts: usize| -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        if data.is_empty() {
            Ok(None)
        } else {
            Ok(Some(data.into_pyarray(py).reshape([parts, n])?))
        }
    };
    let v_grad_m = if values.v_grad_m.is_empty() {
        None
    } else {
        Some(values.v_grad_m.into_pyarray(py).reshape([3, 3, n])?)
    };
    let v_tau = if values.v_tau.is_empty() {
        None
    } else {
        Some(values.v_tau.into_pyarray(py))
    };
    Ok((
        values.energy_density.into_pyarray(py),
        values.v_rho.into_pyarray(py),
        values.v_m.into_pyarray(py).reshape([3, n])?,
        rows(values.v_grad_rho, 3)?,
        v_grad_m,
        v_tau,
        rows(values.v_tau_m, 3)?,
        values.unconverged,
    ))
}

/// The sampling of the sphere: a fixed rule when its directions and
/// weights are given, else the adaptive aligned rules.
fn sampling_from<'a>(
    directions: Option<&'a PyReadonlyArray2<'a, f64>>,
    direction_weights: Option<&'a PyReadonlyArray1<'a, f64>>,
    precision: f64,
    max_level: usize,
) -> PyResult<xc::Sampling<'a>> {
    match (directions, direction_weights) {
        (None, None) => Ok(xc::Sampling::Adaptive {
            precision,
            max_level,
        }),
        (Some(directions), Some(direction_weights)) => {
            let direction_weights = direction_weights.as_slice()?;
            if directions.shape() != [direction_weights.len(), 3] {
                return Err(PyValueError::new_err(
                    "directions have shape [k, 3] with one weight per direction",
                ));
            }
            Ok(xc::Sampling::Fixed(xc::SphereRule {
                directions: directions.as_slice()?,
                weights: direction_weights,
            }))
        }
        _ => Err(PyValueError::new_err(
            "a fixed rule gives its directions and their weights together",
        )),
    }
}

/// The contiguous data of an optional `[parts, n]` array, empty for `None`.
fn to_rows<'a>(
    array: Option<&'a PyReadonlyArray2<'a, f64>>,
    parts: usize,
    n: usize,
    name: &str,
) -> PyResult<&'a [f64]> {
    match array {
        None => Ok(&[]),
        Some(array) => {
            if array.shape() != [parts, n] {
                return Err(PyValueError::new_err(format!(
                    "{name} have shape [{parts}, g] over the same points as the densities"
                )));
            }
            Ok(array.as_slice()?)
        }
    }
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
    module.add_function(wrap_pyfunction!(basis_values, module)?)?;
    module.add_function(wrap_pyfunction!(bloch_basis_values, module)?)?;
    module.add_function(wrap_pyfunction!(partition_weights, module)?)?;
    module.add_function(wrap_pyfunction!(functional_info, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_functional, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_noncollinear, module)?)?;
    module.add_function(wrap_pyfunction!(configure_threads, module)?)?;
    Ok(())
}
