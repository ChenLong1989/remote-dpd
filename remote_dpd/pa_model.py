"""Complex memory-polynomial forward models for model-based DPD learning.

The derivatives in this module use the real inner product on complex vectors,
``real(vdot(left, right))``.  A model linearization is therefore real-linear
and generally contains both a tangent and its complex conjugate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


_COMPLEX_DTYPE = np.dtype(np.complex128)
_FLOAT_DTYPE = np.dtype(np.float64)


@dataclass(frozen=True, slots=True)
class PAForwardModelConfig:
    """Validated settings for ridge fitting a memory-polynomial PA model."""

    orders: tuple[int, ...] = (1, 3, 5, 7, 9)
    memory_depth: int = 3
    ridge: float = 1e-6
    envelope_quantile: float = 0.999
    block_size: int = 256
    validation_every: int = 5
    minimum_scale: float = 1e-12
    column_rms_epsilon: float = 1e-14
    max_condition_number: float = 1e12
    max_validation_nmse_db: float | None = 0.0
    numeric_dtype: str = "complex128"

    def __post_init__(self) -> None:
        orders = _validated_orders(self.orders)
        object.__setattr__(self, "orders", orders)
        if isinstance(self.memory_depth, bool) or not isinstance(self.memory_depth, int):
            raise TypeError("memory_depth must be an integer")
        if self.memory_depth < 1:
            raise ValueError("memory_depth must be at least one")
        ridge = _finite_real_scalar(self.ridge, "ridge")
        if ridge < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        envelope_quantile = _finite_real_scalar(
            self.envelope_quantile,
            "envelope_quantile",
        )
        if not 0.0 < envelope_quantile <= 1.0:
            raise ValueError("envelope_quantile must be in (0, 1]")
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int):
            raise TypeError("block_size must be an integer")
        if self.block_size < 1:
            raise ValueError("block_size must be at least one")
        if isinstance(self.validation_every, bool) or not isinstance(self.validation_every, int):
            raise TypeError("validation_every must be an integer")
        if self.validation_every < 2:
            raise ValueError("validation_every must be at least two")
        minimum_scale = _finite_real_scalar(self.minimum_scale, "minimum_scale")
        if minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be finite and positive")
        column_rms_epsilon = _finite_real_scalar(
            self.column_rms_epsilon,
            "column_rms_epsilon",
        )
        if column_rms_epsilon <= 0.0:
            raise ValueError("column_rms_epsilon must be finite and positive")
        max_condition_number = _finite_real_scalar(
            self.max_condition_number,
            "max_condition_number",
        )
        if max_condition_number < 1.0:
            raise ValueError("max_condition_number must be finite and at least one")
        if self.max_validation_nmse_db is not None:
            _finite_real_scalar(self.max_validation_nmse_db, "max_validation_nmse_db")
        if self.numeric_dtype not in {"complex64", "complex128"}:
            raise ValueError("numeric_dtype must be 'complex64' or 'complex128'")

    @property
    def coefficient_count(self) -> int:
        return len(self.orders) * self.memory_depth


@dataclass(frozen=True, slots=True)
class PAForwardModelDiagnostics:
    """Fit quality and failure information suitable for experiment metrics."""

    train_nmse_db: float
    validation_nmse_db: float
    rank: int
    condition_number: float
    sample_count: int
    train_sample_count: int
    validation_sample_count: int
    coefficient_count: int
    envelope_scale: float
    coefficient_norm: float
    fallback_reason: str | None

    @property
    def succeeded(self) -> bool:
        return self.fallback_reason is None

    def as_metrics(self) -> dict[str, int | float | str | None]:
        """Return names used by the numerical algorithm and experiment logs."""

        return {
            "pa_model_train_nmse_db": self.train_nmse_db,
            "pa_model_validation_nmse_db": self.validation_nmse_db,
            "pa_model_rank": self.rank,
            "pa_model_condition": self.condition_number,
            "pa_model_sample_count": self.sample_count,
            "pa_model_train_sample_count": self.train_sample_count,
            "pa_model_validation_sample_count": self.validation_sample_count,
            "pa_model_coefficient_count": self.coefficient_count,
            "pa_model_envelope_scale": self.envelope_scale,
            "pa_model_coefficient_norm": self.coefficient_norm,
            "pa_model_fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True, slots=True)
class PAForwardModelFit:
    """Result of a fit attempt; failed attempts intentionally contain no model."""

    model: MemoryPolynomialModel | None
    diagnostics: PAForwardModelDiagnostics

    @property
    def succeeded(self) -> bool:
        return self.model is not None and self.diagnostics.succeeded


@dataclass(frozen=True, slots=True, eq=False)
class MemoryPolynomialLinearization:
    """An immutable real-linear model derivative at one fixed input vector.

    For delay ``m``, the local derivative is
    ``a[m] * roll(h, m) + b[m] * conj(roll(h, m))``.  The VJP applies the
    exact inverse circular roll.
    """

    a_coefficients: np.ndarray
    b_coefficients: np.ndarray

    def __post_init__(self) -> None:
        numeric_dtype = _common_complex_dtype(self.a_coefficients, self.b_coefficients)
        a_coefficients = np.asarray(self.a_coefficients, dtype=numeric_dtype)
        b_coefficients = np.asarray(self.b_coefficients, dtype=numeric_dtype)
        if a_coefficients.ndim != 2 or b_coefficients.ndim != 2:
            raise ValueError("linearization coefficients must be two-dimensional")
        if a_coefficients.shape != b_coefficients.shape:
            raise ValueError("linearization coefficient shapes must match")
        if a_coefficients.shape[0] < 1:
            raise ValueError("linearization must contain at least one memory tap")
        if not _is_finite_complex(a_coefficients) or not _is_finite_complex(b_coefficients):
            raise ValueError("linearization coefficients must be finite")
        object.__setattr__(self, "a_coefficients", _immutable_array(a_coefficients, numeric_dtype))
        object.__setattr__(self, "b_coefficients", _immutable_array(b_coefficients, numeric_dtype))

    @property
    def numeric_dtype(self) -> str:
        return self.a_coefficients.dtype.name

    @property
    def input_size(self) -> int:
        return int(self.a_coefficients.shape[1])

    @property
    def memory_depth(self) -> int:
        return int(self.a_coefficients.shape[0])

    def jvp(self, tangent: Sequence[complex] | np.ndarray) -> np.ndarray:
        """Apply the analytic real-linear Jacobian to ``tangent``."""

        tangent_vector = _finite_complex_vector(
            tangent,
            "tangent",
            dtype=self.a_coefficients.dtype,
        )
        if tangent_vector.size != self.input_size:
            raise ValueError("tangent length must match the linearization input length")
        result = np.zeros(self.input_size, dtype=self.a_coefficients.dtype)
        for delay in range(self.memory_depth):
            shifted = np.roll(tangent_vector, delay)
            result += (
                self.a_coefficients[delay] * shifted
                + self.b_coefficients[delay] * np.conjugate(shifted)
            )
        if not _is_finite_complex(result):
            raise FloatingPointError("JVP produced non-finite values")
        return result

    def vjp(self, cotangent: Sequence[complex] | np.ndarray) -> np.ndarray:
        """Apply the adjoint under ``real(vdot(left, right))``."""

        cotangent_vector = _finite_complex_vector(
            cotangent,
            "cotangent",
            dtype=self.a_coefficients.dtype,
        )
        if cotangent_vector.size != self.input_size:
            raise ValueError("cotangent length must match the linearization output length")
        result = np.zeros(self.input_size, dtype=self.a_coefficients.dtype)
        for delay in range(self.memory_depth):
            local_adjoint = (
                np.conjugate(self.a_coefficients[delay]) * cotangent_vector
                + self.b_coefficients[delay] * np.conjugate(cotangent_vector)
            )
            result += np.roll(local_adjoint, -delay)
        if not _is_finite_complex(result):
            raise FloatingPointError("VJP produced non-finite values")
        return result


@dataclass(frozen=True, slots=True, eq=False)
class MemoryPolynomialModel:
    """Immutable complex-coefficient circular memory-polynomial model."""

    orders: tuple[int, ...]
    coefficients: np.ndarray
    envelope_scale: float

    def __post_init__(self) -> None:
        orders = _validated_orders(self.orders)
        numeric_dtype = _complex_dtype(self.coefficients)
        coefficients = np.asarray(self.coefficients, dtype=numeric_dtype)
        if coefficients.ndim != 2:
            raise ValueError("coefficients must have shape (order_count, memory_depth)")
        if coefficients.shape[0] != len(orders) or coefficients.shape[1] < 1:
            raise ValueError("coefficient shape does not match orders or memory depth")
        if not _is_finite_complex(coefficients):
            raise ValueError("coefficients must be finite")
        if not np.isfinite(self.envelope_scale) or self.envelope_scale <= 0.0:
            raise ValueError("envelope_scale must be finite and positive")
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "coefficients", _immutable_array(coefficients, numeric_dtype))
        object.__setattr__(self, "envelope_scale", float(self.envelope_scale))

    @property
    def numeric_dtype(self) -> str:
        return self.coefficients.dtype.name

    @property
    def memory_depth(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def coefficient_count(self) -> int:
        return int(self.coefficients.size)

    def predict(self, input_signal: Sequence[complex] | np.ndarray) -> np.ndarray:
        """Evaluate the model with the coefficient arithmetic dtype."""

        input_vector = _finite_complex_vector(
            input_signal,
            "input_signal",
            dtype=self.coefficients.dtype,
        )
        result = np.zeros(input_vector.size, dtype=self.coefficients.dtype)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for order_index, order in enumerate(self.orders):
                exponent = order - 1
                for delay in range(self.memory_depth):
                    delayed = np.roll(input_vector, delay)
                    radial = (np.abs(delayed) / self.envelope_scale) ** exponent
                    result += self.coefficients[order_index, delay] * delayed * radial
        if not _is_finite_complex(result):
            raise FloatingPointError("model prediction produced non-finite values")
        return result

    def forward(self, input_signal: Sequence[complex] | np.ndarray) -> np.ndarray:
        """Alias for :meth:`predict` for forward-model consumers."""

        return self.predict(input_signal)

    def linearize(self, input_signal: Sequence[complex] | np.ndarray) -> MemoryPolynomialLinearization:
        """Build and freeze the analytic derivative at ``input_signal``."""

        input_vector = _finite_complex_vector(
            input_signal,
            "input_signal",
            dtype=self.coefficients.dtype,
        )
        a_coefficients = np.zeros(
            (self.memory_depth, input_vector.size),
            dtype=self.coefficients.dtype,
        )
        b_coefficients = np.zeros_like(a_coefficients)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for delay in range(self.memory_depth):
                delayed = np.roll(input_vector, delay)
                scaled_magnitude = np.abs(delayed) / self.envelope_scale
                scaled_value = delayed / self.envelope_scale
                for order_index, order in enumerate(self.orders):
                    exponent = order - 1
                    coefficient = self.coefficients[order_index, delay]
                    radial = scaled_magnitude**exponent
                    a_coefficients[delay] += coefficient * (1.0 + 0.5 * exponent) * radial
                    if exponent:
                        b_coefficients[delay] += (
                            coefficient
                            * (0.5 * exponent)
                            * scaled_value**2
                            * scaled_magnitude ** (exponent - 2)
                        )
        if not _is_finite_complex(a_coefficients) or not _is_finite_complex(b_coefficients):
            raise FloatingPointError("model linearization produced non-finite values")
        return MemoryPolynomialLinearization(a_coefficients, b_coefficients)

    def jvp(
        self,
        input_signal: Sequence[complex] | np.ndarray,
        tangent: Sequence[complex] | np.ndarray,
    ) -> np.ndarray:
        """Apply the analytic JVP at ``input_signal``."""

        return self.linearize(input_signal).jvp(tangent)

    def vjp(
        self,
        input_signal: Sequence[complex] | np.ndarray,
        cotangent: Sequence[complex] | np.ndarray,
    ) -> np.ndarray:
        """Apply the analytic real-adjoint VJP at ``input_signal``."""

        return self.linearize(input_signal).vjp(cotangent)


def deterministic_block_split(
    sample_count: int,
    block_size: int = 256,
    validation_every: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train and validation masks for contiguous blocks.

    Blocks are numbered from one.  Every ``validation_every``-th block is held
    out in full; all other blocks are used for fitting.
    """

    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)):
        raise TypeError("sample_count must be an integer")
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise TypeError("block_size must be an integer")
    if block_size < 1:
        raise ValueError("block_size must be at least one")
    if isinstance(validation_every, bool) or not isinstance(validation_every, (int, np.integer)):
        raise TypeError("validation_every must be an integer")
    if validation_every < 2:
        raise ValueError("validation_every must be at least two")
    block_numbers = np.arange(sample_count, dtype=np.int64) // int(block_size) + 1
    validation_mask = block_numbers % int(validation_every) == 0
    train_mask = ~validation_mask
    return train_mask, validation_mask


def fit_pa_model(
    input_signal: Sequence[complex] | np.ndarray,
    output_signal: Sequence[complex] | np.ndarray,
    config: PAForwardModelConfig | None = None,
) -> PAForwardModelFit:
    """Fit a memory polynomial with normalized columns and augmented ridge LS.

    The validation blocks never enter the least-squares system.  Numerical or
    data-quality failures return diagnostics and ``model=None`` so callers can
    perform an explicit safe fallback.
    """

    config = PAForwardModelConfig() if config is None else config
    if not isinstance(config, PAForwardModelConfig):
        raise TypeError("config must be a PAForwardModelConfig")
    numeric_dtype = np.dtype(config.numeric_dtype)
    input_vector = _complex_vector_without_finite_check(input_signal, dtype=numeric_dtype)
    output_vector = _complex_vector_without_finite_check(output_signal, dtype=numeric_dtype)
    sample_count = int(input_vector.size)
    coefficient_count = config.coefficient_count
    if input_vector.size != output_vector.size:
        return _failed_fit(
            "length_mismatch",
            sample_count=sample_count,
            coefficient_count=coefficient_count,
        )

    train_mask, validation_mask = deterministic_block_split(
        sample_count,
        config.block_size,
        config.validation_every,
    )
    train_count = int(np.count_nonzero(train_mask))
    validation_count = int(np.count_nonzero(validation_mask))
    diagnostic_counts = {
        "sample_count": sample_count,
        "train_sample_count": train_count,
        "validation_sample_count": validation_count,
        "coefficient_count": coefficient_count,
    }
    if sample_count == 0:
        return _failed_fit("insufficient_samples", **diagnostic_counts)
    if not _is_finite_complex(input_vector):
        return _failed_fit("non_finite_input", **diagnostic_counts)
    if not _is_finite_complex(output_vector):
        return _failed_fit("non_finite_output", **diagnostic_counts)
    if train_count < coefficient_count:
        return _failed_fit("insufficient_training_samples", **diagnostic_counts)
    if validation_count == 0:
        return _failed_fit("insufficient_validation_samples", **diagnostic_counts)

    envelope_scale = max(
        float(np.quantile(np.abs(input_vector[train_mask]), config.envelope_quantile)),
        config.minimum_scale,
    )
    if not np.isfinite(envelope_scale):
        return _failed_fit("invalid_envelope_scale", **diagnostic_counts)
    design = _design_matrix(input_vector, config.orders, config.memory_depth, envelope_scale)
    if not _is_finite_complex(design):
        return _failed_fit(
            "non_finite_design_matrix",
            envelope_scale=envelope_scale,
            **diagnostic_counts,
        )
    train_design = design[train_mask]
    column_rms = np.sqrt(np.mean(np.abs(train_design) ** 2, axis=0))
    if not np.all(np.isfinite(column_rms)) or np.any(column_rms <= config.column_rms_epsilon):
        return _failed_fit(
            "zero_design_column",
            envelope_scale=envelope_scale,
            **diagnostic_counts,
        )
    normalized_train_design = train_design / column_rms

    try:
        singular_values = np.linalg.svd(normalized_train_design, compute_uv=False)
    except np.linalg.LinAlgError:
        return _failed_fit(
            "singular_value_decomposition_failed",
            envelope_scale=envelope_scale,
            **diagnostic_counts,
        )
    rank, condition_number = _rank_and_condition(singular_values, normalized_train_design.shape)
    rank_fields = {
        "rank": rank,
        "condition_number": condition_number,
        "envelope_scale": envelope_scale,
        **diagnostic_counts,
    }
    if rank < coefficient_count:
        return _failed_fit("rank_deficient", **rank_fields)
    if condition_number > config.max_condition_number:
        return _failed_fit("condition_number_exceeded", **rank_fields)

    fit_design = normalized_train_design
    fit_target = output_vector[train_mask]
    if config.ridge > 0.0:
        fit_design = np.vstack(
            (
                fit_design,
                np.sqrt(config.ridge) * np.eye(coefficient_count, dtype=numeric_dtype),
            )
        )
        fit_target = np.concatenate((fit_target, np.zeros(coefficient_count, dtype=numeric_dtype)))
    try:
        normalized_coefficients, _, _, _ = np.linalg.lstsq(fit_design, fit_target, rcond=None)
    except np.linalg.LinAlgError:
        return _failed_fit("least_squares_failed", **rank_fields)
    coefficients = np.asarray(normalized_coefficients / column_rms, dtype=numeric_dtype)
    if not _is_finite_complex(coefficients):
        return _failed_fit("non_finite_coefficients", **rank_fields)

    coefficient_matrix = coefficients.reshape(len(config.orders), config.memory_depth)
    model = MemoryPolynomialModel(config.orders, coefficient_matrix, envelope_scale)
    try:
        prediction = model.predict(input_vector)
    except FloatingPointError:
        return _failed_fit(
            "non_finite_prediction",
            coefficient_norm=float(np.linalg.norm(coefficients)),
            **rank_fields,
        )
    train_nmse_db = _nmse_db(output_vector[train_mask], prediction[train_mask])
    validation_nmse_db = _nmse_db(output_vector[validation_mask], prediction[validation_mask])
    coefficient_norm = float(np.linalg.norm(coefficients))
    diagnostics = PAForwardModelDiagnostics(
        train_nmse_db=train_nmse_db,
        validation_nmse_db=validation_nmse_db,
        coefficient_norm=coefficient_norm,
        fallback_reason=None,
        **rank_fields,
    )
    if np.isnan(validation_nmse_db) or (
        config.max_validation_nmse_db is not None
        and validation_nmse_db > config.max_validation_nmse_db
    ):
        diagnostics = PAForwardModelDiagnostics(
            train_nmse_db=train_nmse_db,
            validation_nmse_db=validation_nmse_db,
            coefficient_norm=coefficient_norm,
            fallback_reason="validation_nmse_exceeded",
            **rank_fields,
        )
        return PAForwardModelFit(model=None, diagnostics=diagnostics)
    return PAForwardModelFit(model=model, diagnostics=diagnostics)


def fit_memory_polynomial(
    input_signal: Sequence[complex] | np.ndarray,
    output_signal: Sequence[complex] | np.ndarray,
    config: PAForwardModelConfig | None = None,
) -> PAForwardModelFit:
    """Compatibility alias with a model-specific name."""

    return fit_pa_model(input_signal, output_signal, config)


def explicit_real_jacobian(
    model: MemoryPolynomialModel,
    input_signal: Sequence[complex] | np.ndarray,
) -> np.ndarray:
    """Materialize the ``2N x 2N`` real Jacobian for small validation cases."""

    if not isinstance(model, MemoryPolynomialModel):
        raise TypeError("model must be a MemoryPolynomialModel")
    input_vector = _finite_complex_vector(
        input_signal,
        "input_signal",
        dtype=model.coefficients.dtype,
    )
    linearization = model.linearize(input_vector)
    sample_count = input_vector.size
    jacobian = np.empty((2 * sample_count, 2 * sample_count), dtype=np.float64)
    basis = np.zeros(sample_count, dtype=model.coefficients.dtype)
    for index in range(sample_count):
        basis[index] = 1.0
        column = linearization.jvp(basis)
        jacobian[:, index] = _complex_to_real_vector(column)
        basis[index] = 1.0j
        column = linearization.jvp(basis)
        jacobian[:, sample_count + index] = _complex_to_real_vector(column)
        basis[index] = 0.0
    return jacobian


def _design_matrix(
    input_vector: np.ndarray,
    orders: tuple[int, ...],
    memory_depth: int,
    envelope_scale: float,
) -> np.ndarray:
    design = np.empty(
        (input_vector.size, len(orders) * memory_depth),
        dtype=input_vector.dtype,
    )
    column = 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for order in orders:
            exponent = order - 1
            for delay in range(memory_depth):
                delayed = np.roll(input_vector, delay)
                design[:, column] = delayed * (np.abs(delayed) / envelope_scale) ** exponent
                column += 1
    return design


def _rank_and_condition(singular_values: np.ndarray, shape: tuple[int, int]) -> tuple[int, float]:
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0, float("inf")
    tolerance = np.finfo(singular_values.dtype).eps * max(shape) * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > tolerance))
    smallest = float(singular_values[-1])
    condition_number = float(singular_values[0] / smallest) if smallest > 0.0 else float("inf")
    return rank, condition_number


def _failed_fit(
    reason: str,
    *,
    sample_count: int,
    coefficient_count: int,
    train_sample_count: int = 0,
    validation_sample_count: int = 0,
    rank: int = 0,
    condition_number: float = float("inf"),
    envelope_scale: float = float("nan"),
    coefficient_norm: float = float("nan"),
) -> PAForwardModelFit:
    diagnostics = PAForwardModelDiagnostics(
        train_nmse_db=float("nan"),
        validation_nmse_db=float("nan"),
        rank=rank,
        condition_number=condition_number,
        sample_count=sample_count,
        train_sample_count=train_sample_count,
        validation_sample_count=validation_sample_count,
        coefficient_count=coefficient_count,
        envelope_scale=envelope_scale,
        coefficient_norm=coefficient_norm,
        fallback_reason=reason,
    )
    return PAForwardModelFit(model=None, diagnostics=diagnostics)


def _nmse_db(reference: np.ndarray, measured: np.ndarray) -> float:
    error_power = float(np.vdot(measured - reference, measured - reference).real)
    reference_power = float(np.vdot(reference, reference).real)
    if error_power == 0.0:
        return float("-inf")
    if reference_power == 0.0:
        return float("inf")
    return float(10.0 * np.log10(error_power / reference_power))


def _validated_orders(orders: Sequence[int]) -> tuple[int, ...]:
    try:
        values = tuple(orders)
    except TypeError as exc:
        raise TypeError("orders must be a sequence of positive odd integers") from exc
    if not values:
        raise ValueError("orders must not be empty")
    for order in values:
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
            raise TypeError("orders must contain integers")
        if int(order) < 1 or int(order) % 2 == 0:
            raise ValueError("orders must contain positive odd integers")
    normalized = tuple(int(order) for order in values)
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("orders must be unique and strictly increasing")
    return normalized


def _complex_dtype(value: Any) -> np.dtype[Any]:
    return np.dtype(np.complex64) if np.asarray(value).dtype == np.dtype(np.complex64) else _COMPLEX_DTYPE


def _common_complex_dtype(*values: Any) -> np.dtype[Any]:
    return (
        np.dtype(np.complex64)
        if values and all(_complex_dtype(value) == np.dtype(np.complex64) for value in values)
        else _COMPLEX_DTYPE
    )


def _complex_vector_without_finite_check(
    value: Any,
    *,
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    return np.asarray(value, dtype=_complex_dtype(value) if dtype is None else dtype).reshape(-1)


def _finite_complex_vector(
    value: Any,
    name: str,
    *,
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    vector = _complex_vector_without_finite_check(value, dtype=dtype)
    if not _is_finite_complex(vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _is_finite_complex(value: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(value.real)) and np.all(np.isfinite(value.imag)))


def _finite_real_scalar(value: Any, name: str) -> float:
    """Return a finite real scalar while rejecting bools and containers."""

    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _immutable_array(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(contiguous.shape)
    return immutable


def _complex_to_real_vector(value: np.ndarray) -> np.ndarray:
    return np.concatenate((value.real, value.imag)).astype(_FLOAT_DTYPE, copy=False)


__all__ = [
    "MemoryPolynomialLinearization",
    "MemoryPolynomialModel",
    "PAForwardModelConfig",
    "PAForwardModelDiagnostics",
    "PAForwardModelFit",
    "deterministic_block_split",
    "explicit_real_jacobian",
    "fit_memory_polynomial",
    "fit_pa_model",
]
