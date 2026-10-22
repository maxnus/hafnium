"""The subset of the array API that method code may use.

Every ``xp.<name>`` call outside ``hafnium/backend`` must name an entry of
:data:`ALLOWED`. The set is the intersection of what NumPy 2, CuPy and JAX
provide; ``einsum`` is not part of the standard but all three implement
it. ``tests/unit/test_backend_allowed.py`` enforces the rule.
"""

ALLOWED: frozenset[str] = frozenset(
    {
        # creation
        "arange",
        "asarray",
        "eye",
        "full",
        "linspace",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
        # manipulation
        "broadcast_to",
        "concat",
        "expand_dims",
        "flip",
        "moveaxis",
        "permute_dims",
        "reshape",
        "squeeze",
        "stack",
        "take",
        "tile",
        "tril",
        "triu",
        # elementwise
        "abs",
        "clip",
        "conj",
        "cos",
        "exp",
        "floor",
        "imag",
        "isfinite",
        "isinf",
        "isnan",
        "log",
        "maximum",
        "minimum",
        "real",
        "sign",
        "sin",
        "sqrt",
        "where",
        # reductions
        "all",
        "any",
        "argmax",
        "argmin",
        "cumulative_sum",
        "max",
        "mean",
        "min",
        "prod",
        "sum",
        # sorting
        "argsort",
        "sort",
        # linear algebra
        "einsum",
        "linalg.cholesky",
        "linalg.diagonal",
        "linalg.eigh",
        "linalg.inv",
        "linalg.matrix_norm",
        "linalg.matrix_transpose",
        "linalg.qr",
        "linalg.solve",
        "linalg.svd",
        "linalg.trace",
        "linalg.vector_norm",
        "matmul",
        "tensordot",
        "vecdot",
        # dtypes
        "astype",
        "bool",
        "complex128",
        "finfo",
        "float64",
        "iinfo",
        "int64",
        "isdtype",
        "result_type",
    }
)
