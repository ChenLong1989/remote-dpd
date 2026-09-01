"""Versioned runtime contract for device-independent DPD algorithms."""

from __future__ import annotations

import math
import numbers
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np

RUNTIME_API_VERSION = "1.0"


class RuntimeContractError(ValueError):
    """Base error for an invalid runtime request or response."""


class RuntimeConfigurationError(RuntimeContractError):
    """The supplied runtime configuration is invalid or inconsistent."""


class RuntimeInputError(RuntimeContractError):
    """A runtime step input or candidate output violates the contract."""


class RuntimeLifecycleError(RuntimeError):
    """A runtime lifecycle method was called in an invalid state."""


class RuntimeRegistrationError(RuntimeError):
    """A runtime type cannot be added to or resolved from the registry."""


@dataclass(frozen=True, slots=True)
class RuntimeStepInput:
    """All immutable logical inputs needed for one DPD update."""

    x: np.ndarray
    y_current: np.ndarray
    z_current: np.ndarray
    iteration: int
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        x = _copy_signal(self.x, "x")
        y_current = _copy_signal(self.y_current, "y_current")
        z_current = _copy_signal(self.z_current, "z_current")
        if not (x.size == y_current.size == z_current.size):
            raise RuntimeInputError(
                "x, y_current, and z_current must have the same length; "
                f"got {x.size}, {y_current.size}, and {z_current.size}"
            )
        if isinstance(self.iteration, (bool, np.bool_)) or not isinstance(
            self.iteration, numbers.Integral
        ):
            raise RuntimeInputError("iteration must be a non-negative integer")
        iteration = int(self.iteration)
        if iteration < 0:
            raise RuntimeInputError("iteration must be a non-negative integer")

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y_current", y_current)
        object.__setattr__(self, "z_current", z_current)
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "config", _freeze_mapping(self.config, "config"))


@dataclass(frozen=True, slots=True)
class RuntimeStepResult:
    """Candidate transmit waveform and algorithm-owned diagnostic metrics."""

    y_candidate: np.ndarray
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "y_candidate",
            _copy_signal(self.y_candidate, "y_candidate"),
        )
        object.__setattr__(self, "metrics", _freeze_mapping(self.metrics, "metrics"))


class DPDRuntime(ABC):
    """Stable lifecycle and execution boundary for a DPD implementation."""

    api_version: ClassVar[str] = RUNTIME_API_VERSION
    name: ClassVar[str]

    def __init__(self) -> None:
        self._initialized = False
        self._closed = False
        self._config: Mapping[str, Any] | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def config(self) -> Mapping[str, Any]:
        if not self._initialized or self._config is None:
            raise RuntimeLifecycleError("runtime is not initialized")
        return self._config

    def initialize(self, config: Mapping[str, Any]) -> None:
        """Initialize private runtime state from an immutable configuration."""
        if self._closed:
            raise RuntimeLifecycleError("a closed runtime cannot be initialized")
        if self._initialized:
            raise RuntimeLifecycleError("runtime is already initialized")
        prepared = self._prepare_config(_freeze_mapping(config, "config"))
        frozen = _freeze_mapping(prepared, "prepared config")
        self._on_initialize(frozen)
        self._config = frozen
        self._initialized = True

    def step(self, step_input: RuntimeStepInput) -> RuntimeStepResult:
        """Generate one candidate waveform without preprocessing or clipping."""
        if self._closed:
            raise RuntimeLifecycleError("a closed runtime cannot execute a step")
        if not self._initialized or self._config is None:
            raise RuntimeLifecycleError("runtime must be initialized before step")
        if not isinstance(step_input, RuntimeStepInput):
            raise TypeError("step_input must be a RuntimeStepInput")

        step_config = _freeze_mapping(
            self._prepare_config(step_input.config),
            "prepared step config",
        )
        if not _values_equal(self._config, step_config):
            raise RuntimeConfigurationError(
                "step config must match the configuration used to initialize the runtime"
            )
        result = self._step(step_input, self._config)
        if not isinstance(result, RuntimeStepResult):
            raise RuntimeContractError("runtime step must return RuntimeStepResult")
        if result.y_candidate.size != step_input.x.size:
            raise RuntimeInputError(
                "y_candidate must have the same length as x; "
                f"got {result.y_candidate.size} and {step_input.x.size}"
            )
        return result

    def reset(self) -> None:
        """Discard runtime state and require a new initialize call."""
        if self._closed:
            raise RuntimeLifecycleError("a closed runtime cannot be reset")
        self._on_reset()
        self._config = None
        self._initialized = False

    def close(self) -> None:
        """Release runtime resources; repeated close calls are harmless."""
        if self._closed:
            return
        self._on_close()
        self._config = None
        self._initialized = False
        self._closed = True

    def _prepare_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        return config

    def _on_initialize(self, config: Mapping[str, Any]) -> None:
        return None

    def _on_reset(self) -> None:
        return None

    def _on_close(self) -> None:
        return None

    @abstractmethod
    def _step(
        self,
        step_input: RuntimeStepInput,
        config: Mapping[str, Any],
    ) -> RuntimeStepResult:
        raise NotImplementedError


class BasicILCRuntime(DPDRuntime):
    """Basic sample-by-sample ILC with no implicit signal conditioning."""

    name = "basic_ilc"

    def __init__(self) -> None:
        super().__init__()
        self._step_count = 0

    def _prepare_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        unknown = set(config) - {"mu"}
        if unknown:
            raise RuntimeConfigurationError(
                f"unsupported Basic ILC config fields: {sorted(unknown)}"
            )
        mu = config.get("mu", 0.5)
        if isinstance(mu, (bool, np.bool_)) or not isinstance(mu, numbers.Real):
            raise RuntimeConfigurationError("mu must be a finite positive real scalar")
        mu = float(mu)
        if not math.isfinite(mu) or mu <= 0.0:
            raise RuntimeConfigurationError("mu must be a finite positive real scalar")
        return {"mu": mu}

    def _on_initialize(self, config: Mapping[str, Any]) -> None:
        self._step_count = 0

    def _on_reset(self) -> None:
        self._step_count = 0

    def _step(
        self,
        step_input: RuntimeStepInput,
        config: Mapping[str, Any],
    ) -> RuntimeStepResult:
        mu = float(config["mu"])
        error = step_input.z_current - step_input.x
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = step_input.y_current - mu * error
        if not np.all(np.isfinite(candidate)):
            raise RuntimeInputError("Basic ILC produced a non-finite y_candidate")

        self._step_count += 1
        return RuntimeStepResult(
            y_candidate=candidate,
            metrics={
                "iteration": step_input.iteration,
                "runtime_step": self._step_count,
                "mu": mu,
                "error_rms": _rms(error),
                "candidate_rms": _rms(candidate),
            },
        )


_FORWARD_MODEL_DEFAULT_ORDERS: tuple[int, ...] = (1, 3, 5)
_FORWARD_MODEL_DEFAULT_MEMORY_DEPTHS: tuple[int, ...] = (0, 1, 2)
_FORWARD_MODEL_DEFAULT_RIDGE = 1e-8
_FORWARD_MODEL_MAX_ORDER = 9
_FORWARD_MODEL_MAX_MEMORY_DEPTH = 16
_FORWARD_MODEL_MAX_ORDERS = 5
_FORWARD_MODEL_MAX_MEMORY_DEPTHS = 8
_FORWARD_MODEL_MAX_BASIS_TERMS = 16
_FORWARD_MODEL_RIDGE_MIN = 1e-12
_FORWARD_MODEL_RIDGE_MAX = 1e-2
# Upper bound on basis samples held at once while fitting, so the memory cost
# stays bounded for very long waveforms instead of scaling with N * basis size.
_FORWARD_MODEL_BLOCK_TERMS = 8_000_000


@dataclass(frozen=True, slots=True)
class _BasisTerm:
    """One memory-polynomial basis term: y[n-m] * |y[n-m]|**(order-1)."""

    order: int
    memory: int


@dataclass(frozen=True, slots=True)
class _ForwardModelFit:
    """Least-squares fit of the forward model on one (y, z) pair."""

    terms: tuple[_BasisTerm, ...]
    coefficients: np.ndarray
    residual_rms: float


def _basis_column(signal: np.ndarray, term: _BasisTerm) -> np.ndarray:
    """Full-signal periodic basis column for one term."""

    delayed = np.roll(signal, term.memory)
    return delayed * np.abs(delayed) ** (term.order - 1)


def _basis_column_block(
    signal: np.ndarray,
    start: int,
    stop: int,
    term: _BasisTerm,
) -> np.ndarray:
    """Basis column restricted to [start, stop) with whole-signal periodicity."""

    indices = (np.arange(start, stop) - term.memory) % signal.size
    delayed = signal[indices]
    return delayed * np.abs(delayed) ** (term.order - 1)


def _validate_integer_list(
    name: str,
    value: Any,
    *,
    max_items: int,
    minimum: int,
    maximum: int,
    odd_only: bool,
) -> tuple[int, ...]:
    """Validate a strictly ascending bounded integer list."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise RuntimeConfigurationError(f"{name} must be a list of integers")
    if not value:
        raise RuntimeConfigurationError(f"{name} must contain at least one entry")
    if len(value) > max_items:
        raise RuntimeConfigurationError(
            f"{name} must contain at most {max_items} entries; got {len(value)}"
        )
    normalized: list[int] = []
    previous: int | None = None
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Integral):
            raise RuntimeConfigurationError(f"{name} entries must be integers")
        entry = int(item)
        if not minimum <= entry <= maximum or (odd_only and entry % 2 == 0):
            bound = (
                f"positive odd integers up to {maximum}"
                if odd_only
                else f"integers within [{minimum}, {maximum}]"
            )
            raise RuntimeConfigurationError(
                f"{name} entries must be {bound}; got {entry}"
            )
        if previous is not None and entry <= previous:
            raise RuntimeConfigurationError(f"{name} must be strictly ascending")
        normalized.append(entry)
        previous = entry
    return tuple(normalized)


def _fit_memory_polynomial(
    y: np.ndarray,
    z: np.ndarray,
    orders: tuple[int, ...],
    memory_depths: tuple[int, ...],
    ridge: float,
) -> _ForwardModelFit:
    """Fit z ~ sum c_k * basis_k(y) by ridge-regularized complex least squares.

    Columns are RMS-normalized before solving so the relative ridge is
    scale-invariant; the Gram matrix and right-hand side are accumulated
    blockwise to bound peak memory for long waveforms. All-zero columns are
    dropped with a zero coefficient instead of dividing by a zero scale.
    """
    terms = tuple(
        _BasisTerm(order, order_memory)
        for order in orders
        for order_memory in memory_depths
    )
    count = len(terms)
    total = y.size
    gram = np.zeros((count, count), dtype=np.complex128)
    rhs = np.zeros(count, dtype=np.complex128)
    column_energy = np.zeros(count, dtype=np.float64)
    z_energy = float(np.vdot(z, z).real)
    block = max(1, _FORWARD_MODEL_BLOCK_TERMS // count)
    for start in range(0, total, block):
        stop = min(start + block, total)
        z_segment = z[start:stop]
        columns = [_basis_column_block(y, start, stop, term) for term in terms]
        for i in range(count):
            column = columns[i]
            column_energy[i] += float(np.vdot(column, column).real)
            rhs[i] += np.vdot(column, z_segment)
            for j in range(i, count):
                value = np.vdot(column, columns[j])
                gram[i, j] += value
                if j != i:
                    gram[j, i] += np.conj(value)

    scales = np.sqrt(column_energy / total)
    active = np.nonzero(scales > 0.0)[0]
    coefficients = np.zeros(count, dtype=np.complex128)
    if active.size:
        idx = active[:, None], active[None, :]
        reduced = gram[idx] / (scales[active][:, None] * scales[active][None, :])
        reduced[np.diag_indices_from(reduced)] += ridge * total
        reduced_rhs = rhs[active] / scales[active]
        try:
            normalized = np.linalg.solve(reduced, reduced_rhs)
        except np.linalg.LinAlgError:
            normalized = np.linalg.lstsq(reduced, reduced_rhs, rcond=None)[0]
        coefficients[active] = normalized / scales[active]

    model_energy = float(np.vdot(coefficients, gram @ coefficients).real)
    cross_energy = 2.0 * float(np.vdot(coefficients, rhs).real)
    residual_energy = max(0.0, z_energy - cross_energy + model_energy)
    return _ForwardModelFit(
        terms=terms,
        coefficients=coefficients,
        residual_rms=math.sqrt(residual_energy / total),
    )


def _adjoint_gradient(
    y: np.ndarray,
    error: np.ndarray,
    fit: _ForwardModelFit,
) -> np.ndarray:
    """Map the output error to the PA input through the fitted model Jacobian.

    For the non-holomorphic map z = F(y, y*) the steepest-descent direction of
    ||F(y) - x||^2 over the real (Re, Im) parameterization is
    A^H e + conj(B^H e), where A = dF/dy and B = dF/dy*. Both parts are
    required: dropping the conjugated B term tilts the direction enough to
    stall convergence for deeply compressed amplifiers.
    """
    gradient = np.zeros_like(y)
    magnitude = np.abs(y)
    conjugate_squared = np.conj(y) ** 2
    for term, coefficient in zip(fit.terms, fit.coefficients):
        if coefficient == 0.0:
            continue
        order = term.order
        # The derivative weight at input index j uses y[j] itself; the error
        # it pairs with sits at output index j + memory.
        shifted_error = np.roll(error, -term.memory)
        a_weight = (order + 1) / 2.0 * magnitude ** (order - 1)
        gradient += np.conj(coefficient) * a_weight * shifted_error
        if order > 1:
            b_weight = (order - 1) / 2.0 * conjugate_squared * magnitude ** (order - 3)
            gradient += coefficient * np.conj(b_weight * shifted_error)
    return gradient


class ForwardModelILCRuntime(DPDRuntime):
    """ILC update whose direction uses a per-iteration fitted PA forward model.

    The classic sample-wise update treats the PA Jacobian as the identity and
    propagates the feedback error to the PA input unchanged. Under deep gain
    compression that direction is wrong (the local slope approaches zero and
    can turn negative), which slows convergence and eventually diverges. This
    runtime first fits a complex memory-polynomial forward model on the current
    (y, z) pair with a plain least-squares solve, then back-propagates the
    error through the fitted Jacobian and applies the resulting gradient. Model
    accuracy only shapes the direction, so a coarse fit is sufficient.
    """

    name = "forward_model_ilc"

    def __init__(self) -> None:
        super().__init__()
        self._step_count = 0

    def _prepare_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        unknown = set(config) - {"mu", "orders", "memory_depths", "ridge"}
        if unknown:
            raise RuntimeConfigurationError(
                f"unsupported Forward Model ILC config fields: {sorted(unknown)}"
            )
        mu = config.get("mu", 1.0)
        if isinstance(mu, (bool, np.bool_)) or not isinstance(mu, numbers.Real):
            raise RuntimeConfigurationError("mu must be a finite positive real scalar")
        mu = float(mu)
        if not math.isfinite(mu) or mu <= 0.0:
            raise RuntimeConfigurationError("mu must be a finite positive real scalar")

        orders = config.get("orders", _FORWARD_MODEL_DEFAULT_ORDERS)
        memory_depths = config.get(
            "memory_depths", _FORWARD_MODEL_DEFAULT_MEMORY_DEPTHS
        )
        orders_tuple = _validate_integer_list(
            "orders",
            orders,
            max_items=_FORWARD_MODEL_MAX_ORDERS,
            minimum=1,
            maximum=_FORWARD_MODEL_MAX_ORDER,
            odd_only=True,
        )
        memory_tuple = _validate_integer_list(
            "memory_depths",
            memory_depths,
            max_items=_FORWARD_MODEL_MAX_MEMORY_DEPTHS,
            minimum=0,
            maximum=_FORWARD_MODEL_MAX_MEMORY_DEPTH,
            odd_only=False,
        )
        if len(orders_tuple) * len(memory_tuple) > _FORWARD_MODEL_MAX_BASIS_TERMS:
            raise RuntimeConfigurationError(
                "orders x memory_depths must produce at most "
                f"{_FORWARD_MODEL_MAX_BASIS_TERMS} basis terms; "
                f"got {len(orders_tuple) * len(memory_tuple)}"
            )

        ridge = config.get("ridge", _FORWARD_MODEL_DEFAULT_RIDGE)
        if isinstance(ridge, (bool, np.bool_)) or not isinstance(ridge, numbers.Real):
            raise RuntimeConfigurationError("ridge must be a finite real scalar")
        ridge = float(ridge)
        if not math.isfinite(ridge) or not (
            _FORWARD_MODEL_RIDGE_MIN <= ridge <= _FORWARD_MODEL_RIDGE_MAX
        ):
            raise RuntimeConfigurationError(
                f"ridge must be within [{_FORWARD_MODEL_RIDGE_MIN:g}, "
                f"{_FORWARD_MODEL_RIDGE_MAX:g}]"
            )
        return {
            "mu": mu,
            "orders": orders_tuple,
            "memory_depths": memory_tuple,
            "ridge": ridge,
        }

    def _on_initialize(self, config: Mapping[str, Any]) -> None:
        self._step_count = 0

    def _on_reset(self) -> None:
        self._step_count = 0

    def _step(
        self,
        step_input: RuntimeStepInput,
        config: Mapping[str, Any],
    ) -> RuntimeStepResult:
        mu = float(config["mu"])
        orders = tuple(int(value) for value in config["orders"])
        memory_depths = tuple(int(value) for value in config["memory_depths"])
        ridge = float(config["ridge"])
        y = step_input.y_current
        error = step_input.z_current - step_input.x
        with np.errstate(over="ignore", invalid="ignore"):
            fit = _fit_memory_polynomial(
                y, step_input.z_current, orders, memory_depths, ridge
            )
            gradient = _adjoint_gradient(y, error, fit)
            candidate = y - mu * gradient
        if not np.all(np.isfinite(candidate)):
            raise RuntimeInputError(
                "Forward model ILC produced a non-finite y_candidate"
            )

        self._step_count += 1
        return RuntimeStepResult(
            y_candidate=candidate,
            metrics={
                "iteration": step_input.iteration,
                "runtime_step": self._step_count,
                "mu": mu,
                "error_rms": _rms(error),
                "candidate_rms": _rms(candidate),
                "gradient_rms": _rms(gradient),
                "model_residual_rms": fit.residual_rms,
                "model_terms": [
                    {
                        "p": term.order,
                        "m": term.memory,
                        "real": float(coefficient.real),
                        "imag": float(coefficient.imag),
                    }
                    for term, coefficient in zip(fit.terms, fit.coefficients)
                ],
            },
        )


_RUNTIME_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_RUNTIME_TYPES: dict[str, type[DPDRuntime]] = {
    BasicILCRuntime.name: BasicILCRuntime,
    ForwardModelILCRuntime.name: ForwardModelILCRuntime,
}
_REGISTRY_LOCK = RLock()


def register_runtime(
    name: str,
    runtime_type: type[DPDRuntime],
    *,
    replace: bool = False,
) -> None:
    """Register a no-argument runtime class under a normalized name."""
    normalized = _normalize_runtime_name(name)
    if not isinstance(runtime_type, type) or not issubclass(runtime_type, DPDRuntime):
        raise TypeError("runtime_type must be a DPDRuntime subclass")
    if runtime_type.api_version != RUNTIME_API_VERSION:
        raise RuntimeRegistrationError(
            f"runtime API version {runtime_type.api_version!r} is not supported; "
            f"expected {RUNTIME_API_VERSION!r}"
        )
    with _REGISTRY_LOCK:
        if normalized in _RUNTIME_TYPES and not replace:
            raise RuntimeRegistrationError(
                f"runtime {normalized!r} is already registered"
            )
        _RUNTIME_TYPES[normalized] = runtime_type


def create_runtime(name: str) -> DPDRuntime:
    """Create a fresh, uninitialized runtime instance by registered name."""
    normalized = _normalize_runtime_name(name)
    with _REGISTRY_LOCK:
        runtime_type = _RUNTIME_TYPES.get(normalized)
    if runtime_type is None:
        raise RuntimeRegistrationError(
            f"unknown DPD runtime {normalized!r}; available: {list_runtimes()}"
        )
    runtime = runtime_type()
    if not isinstance(runtime, DPDRuntime):
        raise RuntimeRegistrationError(
            f"registered runtime {normalized!r} did not create a DPDRuntime"
        )
    return runtime


def list_runtimes() -> tuple[str, ...]:
    """Return registered runtime names in deterministic order."""
    with _REGISTRY_LOCK:
        return tuple(sorted(_RUNTIME_TYPES))


def _normalize_runtime_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("runtime name must be a string")
    normalized = name.strip().lower()
    if not _RUNTIME_NAME_PATTERN.fullmatch(normalized):
        raise RuntimeRegistrationError(
            "runtime name must start with a letter and contain only "
            "lowercase letters, digits, underscores, or hyphens"
        )
    return normalized


def _copy_signal(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise RuntimeInputError(f"{name} must be a one-dimensional array")
    if raw.size == 0:
        raise RuntimeInputError(f"{name} must not be empty")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise RuntimeInputError(f"{name} must contain numeric samples")
    with np.errstate(over="ignore", invalid="ignore"):
        copied = np.array(raw, dtype=np.complex128, order="C", copy=True)
    if not np.all(np.isfinite(copied)):
        raise RuntimeInputError(f"{name} must contain only finite samples")
    signal = np.frombuffer(copied.tobytes(), dtype=np.complex128).reshape(copied.shape)
    return signal


def _freeze_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeConfigurationError(f"{name} must be a mapping")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise RuntimeConfigurationError(f"{name} keys must be non-empty strings")
        frozen[key] = _freeze_value(item)
    return MappingProxyType(frozen)


def _freeze_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise RuntimeConfigurationError("runtime config numbers must be finite")
        return normalized
    if isinstance(value, numbers.Complex):
        normalized = complex(value)
        if not math.isfinite(normalized.real) or not math.isfinite(normalized.imag):
            raise RuntimeConfigurationError("runtime config numbers must be finite")
        return normalized
    if isinstance(value, Mapping):
        return _freeze_mapping(value, "nested config value")
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise RuntimeConfigurationError(
                "runtime config arrays must not contain Python objects"
            )
        result = np.array(value, order="C", copy=True)
        if np.issubdtype(result.dtype, np.number) and not np.all(np.isfinite(result)):
            raise RuntimeConfigurationError("runtime config arrays must be finite")
        return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    raise RuntimeConfigurationError(
        f"unsupported runtime config value type {type(value).__name__!r}"
    )


def _values_equal(first: Any, second: Any) -> bool:
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return set(first) == set(second) and all(
            _values_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        try:
            return bool(np.array_equal(first, second, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(first, second))
    if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        return len(first) == len(second) and all(
            _values_equal(left, right) for left, right in zip(first, second)
        )
    try:
        result = first == second
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _rms(signal: np.ndarray) -> float:
    magnitude = np.abs(signal)
    scale = float(np.max(magnitude))
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        return scale
    return float(scale * np.sqrt(np.mean((magnitude / scale) ** 2)))
