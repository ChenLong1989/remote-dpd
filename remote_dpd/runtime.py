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


_FORWARD_MODEL_DEFAULT_TAP_COUNT = 17
_FORWARD_MODEL_DEFAULT_LUT_SIZE = 32
# Plain Tikhonov floor: the FLF basis is structurally rank-deficient, so the
# solve always needs a small identity term; the LUT difference penalty is the
# regularization that actually shapes the coefficients.
_FORWARD_MODEL_DEFAULT_RIDGE = 1e-8
_FORWARD_MODEL_DEFAULT_LUT_RIDGE = 1e-3
_FORWARD_MODEL_RIDGE_MIN = 1e-12
_FORWARD_MODEL_RIDGE_MAX = 1e-2
# Upper bound on the complex column count so a misconfigured ladder cannot
# allocate a prohibitive Gram matrix (46 taps x 64 knots -> 5734 columns).
_FORWARD_MODEL_MAX_COLUMNS = 6144


class ForwardModelILCRuntime(DPDRuntime):
    """ILC update whose direction uses a per-iteration fitted PA forward model.

    The classic sample-wise update treats the PA Jacobian as the identity and
    propagates the feedback error to the PA input unchanged. Under deep gain
    compression that direction is wrong (the local slope approaches zero and
    can turn negative), which slows convergence and eventually diverges. This
    runtime instead fits a residual FIR-LUT-FIR forward model
    ``z ~ y + Phi(y) w`` (ported from dpd-compass, see
    ``remote_dpd/forward_model.py``) on the current (y, z) pair with a
    regularized least-squares solve, then back-propagates the error through
    the fitted Jacobian and applies the resulting gradient.

    The identity term is part of the model, not of the basis: the fitted
    Jacobian always contains it, so the ILC fixed point stays at
    ``(I + J_Phi)^H e = 0`` instead of drifting with model mismatch, and the
    update degrades continuously to basic ILC as the basis coefficients
    shrink toward zero.
    """

    name = "forward_model_ilc"

    def __init__(self) -> None:
        super().__init__()
        self._step_count = 0

    @staticmethod
    def _flf_model(tap_count: int, lut_size: int):
        # Imported lazily: forward_model imports RuntimeInputError from this
        # module, so a top-level import would be circular.
        from remote_dpd.forward_model import FLFResidualModel

        return FLFResidualModel(tap_count, lut_size)

    def _prepare_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        unknown = set(config) - {
            "mu",
            "tap_count",
            "lut_size",
            "ridge",
            "lut_ridge",
        }
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

        from remote_dpd.forward_model import TAP_COUNTS

        tap_count = config.get("tap_count", _FORWARD_MODEL_DEFAULT_TAP_COUNT)
        if isinstance(tap_count, (bool, np.bool_)) or not isinstance(tap_count, numbers.Integral):
            raise RuntimeConfigurationError(
                f"tap_count must be one of {TAP_COUNTS}"
            )
        tap_count = int(tap_count)
        if tap_count not in TAP_COUNTS:
            raise RuntimeConfigurationError(
                f"tap_count must be one of {TAP_COUNTS}; got {tap_count}"
            )

        lut_size = config.get("lut_size", _FORWARD_MODEL_DEFAULT_LUT_SIZE)
        if (
            isinstance(lut_size, (bool, np.bool_))
            or not isinstance(lut_size, numbers.Integral)
            or int(lut_size) < 3
        ):
            raise RuntimeConfigurationError(
                f"lut_size must be an integer >= 3; got {lut_size!r}"
            )
        lut_size = int(lut_size)

        columns = self._flf_model(tap_count, lut_size).n_columns()
        if columns > _FORWARD_MODEL_MAX_COLUMNS:
            raise RuntimeConfigurationError(
                "tap_count x lut_size must produce at most "
                f"{_FORWARD_MODEL_MAX_COLUMNS} basis columns; got {columns}"
            )

        validated: dict[str, float] = {}
        for field in ("ridge", "lut_ridge"):
            value = config.get(
                field,
                _FORWARD_MODEL_DEFAULT_RIDGE
                if field == "ridge"
                else _FORWARD_MODEL_DEFAULT_LUT_RIDGE,
            )
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                raise RuntimeConfigurationError(
                    f"{field} must be a finite real scalar"
                )
            value = float(value)
            if not math.isfinite(value) or not (
                _FORWARD_MODEL_RIDGE_MIN <= value <= _FORWARD_MODEL_RIDGE_MAX
            ):
                raise RuntimeConfigurationError(
                    f"{field} must be within "
                    f"[{_FORWARD_MODEL_RIDGE_MIN:g}, {_FORWARD_MODEL_RIDGE_MAX:g}]"
                )
            validated[field] = value

        return {
            "mu": mu,
            "tap_count": tap_count,
            "lut_size": lut_size,
            "ridge": validated["ridge"],
            "lut_ridge": validated["lut_ridge"],
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
        model = self._flf_model(int(config["tap_count"]), int(config["lut_size"]))
        y = step_input.y_current
        error = step_input.z_current - step_input.x
        with np.errstate(over="ignore", invalid="ignore"):
            fit = model.fit(
                y, step_input.z_current, float(config["ridge"]), float(config["lut_ridge"])
            )
            gradient = model.adjoint(y, error, fit.coefficients)
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
                "model_coefficients": _coefficient_summary(model, fit),
            },
        )


def _coefficient_summary(model, fit) -> Mapping[str, Any]:
    """Compact fit summary: full alpha block plus beta magnitude statistics.

    The beta block holds up to a few thousand complex numbers, which is too
    heavy for per-iteration JSON metrics, so it is reduced to moments.
    """
    coefficients = fit.coefficients
    linear = 2 * model.linear_count
    alpha = coefficients[:linear]
    beta = coefficients[linear:]
    beta_magnitudes = np.abs(beta)
    return {
        "tap_count": model.tap_count,
        "lut_size": model.lut_size,
        "columns": int(coefficients.size),
        "alpha": [
            {
                "phase": phase,
                "delay": delay,
                "real": float(value.real),
                "imag": float(value.imag),
            }
            for phase in (0, 1)
            for delay, value in zip(model.linear_taps, alpha[model.linear_count * phase : model.linear_count * (phase + 1)])
        ],
        "beta_count": int(beta.size),
        "beta_rms": float(np.sqrt(np.mean(beta_magnitudes**2))) if beta.size else 0.0,
        "beta_max": float(np.max(beta_magnitudes)) if beta.size else 0.0,
    }


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
