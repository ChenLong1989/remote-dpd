"""Typed configuration and legacy MATLAB struct compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    enable_equalizer: bool = False
    reset: bool = False
    debug: bool = False
    tx_fir: np.ndarray | None = None
    error_fir: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ilc_mu(self) -> float:
        return float(self.learning_rate)


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
    internal_mhz = float(_scalar(raw.get("InternalSamplingRate"), 983.04))
    feedback_mhz = float(_scalar(raw.get("FeedbackSamplingRate", raw.get("FBSamplingRate")), internal_mhz))
    output_mhz = float(_scalar(raw.get("OutputSamplingRate"), internal_mhz))
    bandwidth = float(_scalar(raw.get("BW", raw.get("FB_BW")), 700.0))
    if bandwidth < 1e6:
        bandwidth *= 1e6

    learning_rate = float(_scalar(raw.get("LearningRate"), 0.5))
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        learning_rate = 0.5
    # The historical Zilink ILC path overrides mu to 0.3.
    if supplier.lower() in {"zilink", "zillnk"} and not any(
        key in raw for key in ("LearningRate", "ILCMu", "mu")
    ):
        learning_rate = 0.3
    else:
        learning_rate = float(_scalar(raw.get("ILCMu", raw.get("mu")), learning_rate))

    starting_sample = max(1, int(_scalar(raw.get("StartingSample"), 1)))
    phase_compensate = _bool(
        raw.get("phaseCompensate", raw.get("phase_compensate")),
        supplier.lower() in {"zilink", "zillnk"},
    )
    threshold = float(_scalar(raw.get("phaseCompThr", raw.get("phaseCompensationThreshold")), 0.15))
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = 0.15

    def taps(*keys: str) -> np.ndarray | None:
        value = first_value(raw, *keys)
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        return array if array.size > 1 else None

    known = {
        "configDPD", "supplierName", "InternalSamplingRate", "FeedbackSamplingRate",
        "FBSamplingRate", "OutputSamplingRate", "BW", "FB_BW", "LearningRate",
        "ILCMu", "mu", "StartingSample", "phaseCompensate", "phase_compensate",
        "phaseCompThr", "phaseCompensationThreshold", "PAPR", "enableEq", "debug",
        "Reset", "txFirHd", "errFirHd", "run_idealDPD", "enILC", "idealDPD",
    }
    return LegacyConfig(
        supplier_name=supplier,
        sample_rate_hz=internal_mhz * 1e6,
        feedback_sample_rate_hz=feedback_mhz * 1e6,
        output_sample_rate_hz=output_mhz * 1e6,
        bandwidth_hz=bandwidth,
        learning_rate=learning_rate,
        alpha=float(_scalar(raw.get("alpha"), 0.0)),
        dpd_gain_db=float(_scalar(raw.get("dpdGainDb"), 0.0)),
        starting_sample=starting_sample,
        papr_db=float(_scalar(raw.get("PAPR"), 7.5)),
        phase_compensate=phase_compensate,
        phase_compensation_threshold=threshold,
        enable_equalizer=_bool(raw.get("enableEq"), False),
        reset=_bool(raw.get("Reset"), False),
        debug=_bool(raw.get("debug"), False),
        tx_fir=taps("txFirHd", "tx_fir"),
        error_fir=taps("errFirHd", "err_fir"),
        extra={str(key): value for key, value in raw.items() if key not in known},
    )
