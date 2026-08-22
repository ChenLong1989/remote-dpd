"""Typed configuration and legacy MATLAB struct compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping

import numpy as np

from .protocol import first_value


def _scalar(value: Any, default: Any) -> Any:
    if value is None:
        return default
    array = np.asarray(value)
    if array.size == 0:
        return default
    item = array.reshape(-1)[0]
    if isinstance(item, np.generic):
        return item.item()
    return item


def _bool(value: Any, default: bool = False) -> bool:
    value = _scalar(value, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _string(value: Any, default: str) -> str:
    item = _scalar(value, default)
    return str(item).strip().lower() or default


def _finite_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        item = float(_scalar(value, default))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"expected a finite number, got {value!r}") from exc
    if not np.isfinite(item):
        raise ValueError(f"expected a finite number, got {item!r}")
    return item


def _positive_float(value: Any, default: float) -> float:
    item = _finite_float(value, default)
    if item <= 0.0:
        raise ValueError(f"expected a positive number, got {item!r}")
    return item


def _nonnegative_float(value: Any, default: float) -> float:
    item = _finite_float(value, default)
    if item < 0.0:
        raise ValueError(f"expected a non-negative number, got {item!r}")
    return item


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    item = _positive_float(value, 1.0)
    return item


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    try:
        scalar = _scalar(value, default)
        if isinstance(scalar, (bool, np.bool_)):
            raise ValueError(f"expected a positive integer, got {scalar!r}")
        numeric = float(scalar)
        item = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"expected a positive integer, got {value!r}") from exc
    if not np.isfinite(numeric) or numeric != item or item <= 0:
        raise ValueError(f"expected a positive integer, got {scalar!r}")
    return item


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "_fieldnames"):
        return {name: getattr(value, name) for name in value._fieldnames}
    return {}


@dataclass(slots=True)
class LegacyConfig:
    """Normalized settings used by the ILC engine and file service."""

    supplier_name: str = "default"
    sample_rate_hz: float = 983.04e6
    feedback_sample_rate_hz: float = 983.04e6
    output_sample_rate_hz: float = 983.04e6
    bandwidth_hz: float = 700e6
    learning_rate: float = 0.5
    alpha: float = 0.0
    dpd_gain_db: float = 0.0
    starting_sample: int = 1
    papr_db: float = 7.5
    phase_compensate: bool = False
    phase_compensation_threshold: float = 0.15
    ilc_backward_mode: str = "legacy"
    calibration_mode: str = "auto"
    calibration_coefficient: complex | None = None
    pa_model_order: int = 9
    pa_model_memory_depth: int = 3
    pa_model_ridge: float = 1e-6
    pa_model_min_validation_nmse_db: float = -20.0
    ilc_lm_damping: float = 1e-2
    ilc_cg_max_iterations: int = 8
    ilc_cg_tolerance: float = 1e-3
    ilc_trust_region_ratio: float = 0.25
    ilc_max_input_rms: float | None = None
    ilc_max_input_peak: float | None = None
    ilc_max_input_papr_db: float | None = None
    pa_model_fallback: str = "linear"
    enable_equalizer: bool = False
    reset: bool = False
    debug: bool = False
    tx_fir: np.ndarray | None = None
    error_fir: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ilc_mu(self) -> float:
        return float(self.learning_rate)


def algorithm_config_fingerprint(config: LegacyConfig) -> str:
    """Hash settings that change the target, calibration, or update law."""
    digest = hashlib.sha256()
    scalar_fields = (
        "sample_rate_hz",
        "feedback_sample_rate_hz",
        "output_sample_rate_hz",
        "learning_rate",
        "alpha",
        "dpd_gain_db",
        "phase_compensate",
        "phase_compensation_threshold",
        "ilc_backward_mode",
        "calibration_mode",
        "calibration_coefficient",
        "pa_model_order",
        "pa_model_memory_depth",
        "pa_model_ridge",
        "pa_model_min_validation_nmse_db",
        "ilc_lm_damping",
        "ilc_cg_max_iterations",
        "ilc_cg_tolerance",
        "ilc_trust_region_ratio",
        "ilc_max_input_rms",
        "ilc_max_input_peak",
        "ilc_max_input_papr_db",
        "pa_model_fallback",
    )
    for name in scalar_fields:
        digest.update(name.encode("ascii"))
        digest.update(repr(getattr(config, name)).encode("utf-8"))
        digest.update(b"\0")
    for name in ("tx_fir", "error_fir"):
        value = getattr(config, name)
        digest.update(name.encode("ascii"))
        if value is None:
            digest.update(b"none")
            continue
        array = np.ascontiguousarray(np.asarray(value, dtype=np.complex128).reshape(-1))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def config_from_mat(
    payload: Mapping[str, Any] | None,
    *,
    supplier_name: str = "default",
) -> LegacyConfig:
    """Convert `configDPD` from old MAT files to a stable Python config.

    Old release/frequency fields are deliberately retained in `extra` only;
    they no longer select a MARS/MADE engine. The service always uses ILC.
    """
    payload = payload or {}
    raw = _mapping(payload.get("configDPD", payload))
    supplier = str(_scalar(raw.get("supplierName", supplier_name), supplier_name))
    internal_mhz = _positive_float(raw.get("InternalSamplingRate"), 983.04)
    feedback_mhz = _positive_float(
        raw.get("FeedbackSamplingRate", raw.get("FBSamplingRate")),
        internal_mhz,
    )
    output_mhz = _positive_float(raw.get("OutputSamplingRate"), internal_mhz)
    bandwidth = _positive_float(raw.get("BW", raw.get("FB_BW")), 700.0)
    if bandwidth < 1e6:
        bandwidth *= 1e6

    learning_rate = _positive_float(raw.get("LearningRate"), 0.5)
    # The historical Zilink ILC path overrides mu to 0.3.
    if supplier.lower() in {"zilink", "zillnk"} and not any(
        key in raw for key in ("LearningRate", "ILCMu", "mu")
    ):
        learning_rate = 0.3
    else:
        learning_rate = _positive_float(raw.get("ILCMu", raw.get("mu")), learning_rate)

    starting_sample = _positive_int(raw.get("StartingSample"), 1)
    phase_compensate = _bool(
        raw.get("phaseCompensate", raw.get("phase_compensate")),
        supplier.lower() in {"zilink", "zillnk"},
    )
    threshold = _positive_float(
        raw.get("phaseCompThr", raw.get("phaseCompensationThreshold")),
        0.15,
    )

    backward_mode = _string(
        raw.get("ILCBackwardMode", raw.get("ilc_backward_mode")),
        "legacy",
    )
    mode_aliases = {
        "identity": "linear",
        "legacy_ilc": "legacy",
        "linear_ilc": "linear",
        "instantaneous_gain_ilc": "instantaneous_gain",
        "raw_vjp": "model_vjp",
        "vjp": "model_vjp",
        "model_vjp_ilc": "model_vjp",
        "lm_vjp": "model_lm",
        "model_lm_ilc": "model_lm",
    }
    backward_mode = mode_aliases.get(backward_mode, backward_mode)
    if backward_mode not in {"legacy", "linear", "instantaneous_gain", "model_vjp", "model_lm"}:
        raise ValueError(f"unsupported ILC backward mode {backward_mode!r}")

    calibration_mode = _string(
        raw.get("ILCCalibrationMode", raw.get("calibration_mode")),
        "auto",
    )
    if calibration_mode not in {"auto", "legacy_dynamic", "frozen_first", "explicit"}:
        raise ValueError(f"unsupported ILC calibration mode {calibration_mode!r}")
    calibration_value = first_value(
        raw,
        "ILCCalibrationCoefficient",
        "FeedbackCalibration",
        "calibration_coefficient",
    )
    calibration_coefficient: complex | None = None
    if calibration_value is not None:
        candidate = complex(_scalar(calibration_value, 1.0 + 0.0j))
        if not (np.isfinite(candidate.real) and np.isfinite(candidate.imag)):
            raise ValueError("feedback calibration coefficient must be finite")
        if abs(candidate) <= np.finfo(float).eps:
            raise ValueError("feedback calibration coefficient must be non-zero")
        calibration_coefficient = candidate
    if calibration_mode == "explicit" and calibration_coefficient is None:
        raise ValueError("explicit calibration requires ILCCalibrationCoefficient")

    def taps(*keys: str) -> np.ndarray | None:
        value = first_value(raw, *keys)
        if value is None:
            return None
        array = np.asarray(value, dtype=np.complex128).reshape(-1)
        if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
            raise ValueError(f"FIR taps {keys[0]} must be finite")
        return array if array.size > 1 else None

    alpha = _finite_float(raw.get("alpha"), 0.0)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    gain_db = _finite_float(raw.get("dpdGainDb"), 0.0)
    papr_db = _positive_float(raw.get("PAPR"), 7.5)
    model_order = _positive_int(raw.get("PAModelOrder"), 9)
    if model_order % 2 == 0 or model_order > 21:
        raise ValueError("PAModelOrder must be an odd integer in [1, 21]")
    memory_depth = _positive_int(raw.get("PAModelMemoryDepth"), 3)
    if memory_depth > 16:
        raise ValueError("PAModelMemoryDepth must be in [1, 16]")
    ridge = _nonnegative_float(raw.get("PAModelRidge"), 1e-6)
    validation_threshold = _finite_float(raw.get("PAModelMinValidationNmseDb"), -20.0)
    if validation_threshold > 0.0:
        raise ValueError("PAModelMinValidationNmseDb must be at most 0 dB")
    lm_damping = _positive_float(raw.get("ILCLMDamping"), 1e-2)
    if lm_damping < 1e-8:
        raise ValueError("ILCLMDamping must be at least 1e-8")
    cg_iterations = _positive_int(raw.get("ILCCGMaxIterations"), 8)
    if cg_iterations > 128:
        raise ValueError("ILCCGMaxIterations must be in [1, 128]")
    cg_tolerance = _positive_float(raw.get("ILCCGTolerance"), 1e-3)
    if cg_tolerance >= 1.0:
        raise ValueError("ILCCGTolerance must be in (0, 1)")
    trust_region_ratio = _positive_float(raw.get("ILCTrustRegionRatio"), 0.25)
    if trust_region_ratio > 1.0:
        raise ValueError("ILCTrustRegionRatio must be in (0, 1]")
    fallback = _string(raw.get("PAModelFallback"), "linear")
    if fallback not in {"linear", "hold"}:
        raise ValueError("PAModelFallback must be 'linear' or 'hold'")
    tx_fir = taps("txFirHd", "tx_fir")
    error_fir = taps("errFirHd", "err_fir")
    if backward_mode in {"model_vjp", "model_lm"}:
        conflicts = []
        if phase_compensate:
            conflicts.append("phaseCompensate")
        if alpha != 0.0:
            conflicts.append("alpha")
        if tx_fir is not None:
            conflicts.append("txFirHd")
        if error_fir is not None:
            conflicts.append("errFirHd")
        if conflicts:
            joined = ", ".join(conflicts)
            raise ValueError(f"model-based ILC is incompatible with legacy settings: {joined}")

    known = {
        "configDPD", "supplierName", "InternalSamplingRate", "FeedbackSamplingRate",
        "FBSamplingRate", "OutputSamplingRate", "BW", "FB_BW", "LearningRate",
        "ILCMu", "mu", "StartingSample", "phaseCompensate", "phase_compensate",
        "phaseCompThr", "phaseCompensationThreshold", "PAPR", "enableEq", "debug",
        "Reset", "txFirHd", "errFirHd", "tx_fir", "err_fir", "alpha",
        "dpdGainDb", "run_idealDPD", "enILC", "idealDPD", "ILCBackwardMode",
        "ilc_backward_mode", "ILCCalibrationMode", "calibration_mode",
        "ILCCalibrationCoefficient", "FeedbackCalibration", "calibration_coefficient",
        "PAModelType", "PAModelOrder", "PAModelMemoryDepth", "PAModelRidge",
        "PAModelMinValidationNmseDb", "ILCLMDamping", "ILCCGMaxIterations",
        "ILCCGTolerance", "ILCTrustRegionRatio", "ILCMaxInputRms",
        "ILCMaxInputPeak", "ILCMaxInputPaprDb", "PAModelFallback",
    }
    return LegacyConfig(
        supplier_name=supplier,
        sample_rate_hz=internal_mhz * 1e6,
        feedback_sample_rate_hz=feedback_mhz * 1e6,
        output_sample_rate_hz=output_mhz * 1e6,
        bandwidth_hz=bandwidth,
        learning_rate=learning_rate,
        alpha=alpha,
        dpd_gain_db=gain_db,
        starting_sample=starting_sample,
        papr_db=papr_db,
        phase_compensate=phase_compensate,
        phase_compensation_threshold=threshold,
        ilc_backward_mode=backward_mode,
        calibration_mode=calibration_mode,
        calibration_coefficient=calibration_coefficient,
        pa_model_order=model_order,
        pa_model_memory_depth=memory_depth,
        pa_model_ridge=ridge,
        pa_model_min_validation_nmse_db=validation_threshold,
        ilc_lm_damping=lm_damping,
        ilc_cg_max_iterations=cg_iterations,
        ilc_cg_tolerance=cg_tolerance,
        ilc_trust_region_ratio=trust_region_ratio,
        ilc_max_input_rms=_optional_positive_float(raw.get("ILCMaxInputRms")),
        ilc_max_input_peak=_optional_positive_float(raw.get("ILCMaxInputPeak")),
        ilc_max_input_papr_db=_optional_positive_float(raw.get("ILCMaxInputPaprDb")),
        pa_model_fallback=fallback,
        enable_equalizer=_bool(raw.get("enableEq"), False),
        reset=_bool(raw.get("Reset"), False),
        debug=_bool(raw.get("debug"), False),
        tx_fir=tx_fir,
        error_fir=error_fir,
        extra={str(key): value for key, value in raw.items() if key not in known},
    )
