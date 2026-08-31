"""Bounded RF analysis for the trusted-network Web console."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ANALYSIS_SCHEMA_VERSION = 1
MIN_ANALYSIS_POINTS = 256
MAX_ANALYSIS_POINTS = 4096
MAX_MEASUREMENT_BANDS = 32
MAX_BAND_LABEL_LENGTH = 48
MAX_ANALYSIS_TRACES = 4
MAX_CACHE_ENTRIES = 16
MAX_CACHE_BYTES = 16 * 1024 * 1024
DEFAULT_AMPLITUDE_FLOOR_DB = -50.0
AM_BINS = 64
MAX_STIMULUS_RESPONSE_SAMPLES = 262_144
SPECTRUM_FLOOR_DBFS = -300.0
_CHUNK_SAMPLES = 1_000_000
_TRACE_NAMES = frozenset({"baseline_z", "target_z", "reference_x", "target_y"})
_BAND_ROLES = frozenset({"main", "adjacent", "other"})
_FREQUENCY_MODES = frozenset({"relative", "absolute"})


class WebAnalysisError(ValueError):
    """Stable RF-analysis error with an explicit HTTP status."""

    def __init__(self, code: str, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class MeasurementBand:
    """One frequency band integrated from the complete periodic DFT."""

    label: str
    role: str
    center_offset_hz: float
    integration_bandwidth_hz: float
    enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "role": self.role,
            "center_offset_hz": self.center_offset_hz,
            "integration_bandwidth_hz": self.integration_bandwidth_hz,
            "enabled": self.enabled,
        }


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """Strict, cacheable selection and display profile for one analysis call."""

    baseline_iteration: int | None
    target_iteration: int | None
    points: int
    frequency_mode: str
    traces: tuple[str, ...]
    bands: tuple[MeasurementBand, ...]
    amplitude_floor_db: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AnalysisRequest:
        if not isinstance(payload, Mapping):
            raise WebAnalysisError(
                "invalid_analysis_request",
                "analysis request must be a JSON object",
            )
        allowed = {
            "schema_version",
            "baseline_iteration",
            "target_iteration",
            "points",
            "frequency_mode",
            "traces",
            "bands",
            "amplitude_floor_db",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise WebAnalysisError(
                "invalid_analysis_request",
                f"unsupported analysis fields: {sorted(unknown)}",
            )
        schema_version = _integer(
            payload.get("schema_version", ANALYSIS_SCHEMA_VERSION),
            "schema_version",
            minimum=ANALYSIS_SCHEMA_VERSION,
            maximum=ANALYSIS_SCHEMA_VERSION,
        )
        if schema_version != ANALYSIS_SCHEMA_VERSION:  # pragma: no cover
            raise WebAnalysisError(
                "unsupported_analysis_schema",
                "analysis schema version is not supported",
            )
        points = _integer(
            payload.get("points", 1600),
            "points",
            minimum=MIN_ANALYSIS_POINTS,
            maximum=MAX_ANALYSIS_POINTS,
        )
        frequency_mode = payload.get("frequency_mode", "relative")
        if (
            not isinstance(frequency_mode, str)
            or frequency_mode not in _FREQUENCY_MODES
        ):
            raise WebAnalysisError(
                "invalid_frequency_mode",
                "frequency_mode must be relative or absolute",
            )
        raw_traces = payload.get("traces", ["baseline_z", "target_z"])
        if not isinstance(raw_traces, list) or not raw_traces:
            raise WebAnalysisError(
                "invalid_analysis_traces",
                "traces must be a non-empty JSON array",
            )
        if len(raw_traces) > MAX_ANALYSIS_TRACES:
            raise WebAnalysisError(
                "analysis_limit_exceeded",
                f"at most {MAX_ANALYSIS_TRACES} traces may be requested",
                status_code=413,
            )
        traces: list[str] = []
        for value in raw_traces:
            if not isinstance(value, str) or value not in _TRACE_NAMES:
                raise WebAnalysisError(
                    "invalid_analysis_traces",
                    f"unsupported analysis trace {value!r}",
                )
            if value in traces:
                raise WebAnalysisError(
                    "invalid_analysis_traces",
                    "analysis traces must be unique",
                )
            traces.append(value)
        raw_bands = payload.get("bands", [])
        if not isinstance(raw_bands, list):
            raise WebAnalysisError(
                "invalid_measurement_bands",
                "bands must be a JSON array",
            )
        if len(raw_bands) > MAX_MEASUREMENT_BANDS:
            raise WebAnalysisError(
                "analysis_limit_exceeded",
                f"at most {MAX_MEASUREMENT_BANDS} measurement bands are allowed",
                status_code=413,
            )
        bands = tuple(_measurement_band(item) for item in raw_bands)
        normalized_labels = [band.label.casefold() for band in bands]
        if len(set(normalized_labels)) != len(normalized_labels):
            raise WebAnalysisError(
                "invalid_measurement_bands",
                "measurement-band labels must be unique",
            )
        amplitude_floor_db = _finite_real(
            payload.get("amplitude_floor_db", DEFAULT_AMPLITUDE_FLOOR_DB),
            "amplitude_floor_db",
        )
        if not -120.0 <= amplitude_floor_db <= -10.0:
            raise WebAnalysisError(
                "invalid_amplitude_floor",
                "amplitude_floor_db must be between -120 and -10 dB",
            )
        return cls(
            baseline_iteration=_optional_iteration(
                payload.get("baseline_iteration"), "baseline_iteration"
            ),
            target_iteration=_optional_iteration(
                payload.get("target_iteration"), "target_iteration"
            ),
            points=points,
            frequency_mode=frequency_mode,
            traces=tuple(traces),
            bands=bands,
            amplitude_floor_db=amplitude_floor_db,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "baseline_iteration": self.baseline_iteration,
            "target_iteration": self.target_iteration,
            "points": self.points,
            "frequency_mode": self.frequency_mode,
            "traces": list(self.traces),
            "bands": [band.to_dict() for band in self.bands],
            "amplitude_floor_db": self.amplitude_floor_db,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class AnalysisRecord:
    """One immutable, fully evaluated DPD round used for Web analysis."""

    iteration: int
    y: np.ndarray
    z: np.ndarray
    power_dbm: float | None = None
    attenuation_db: float | None = None
    nmse_db: float | None = None


class RFAnalysisEngine:
    """Serialize expensive FFT work and cache only bounded JSON results."""

    def __init__(
        self,
        *,
        max_cache_entries: int = MAX_CACHE_ENTRIES,
        max_cache_bytes: int = MAX_CACHE_BYTES,
    ) -> None:
        self._gate = threading.BoundedSemaphore(1)
        self._cache_lock = threading.Lock()
        self._cache: OrderedDict[tuple[Any, ...], tuple[dict[str, Any], int]] = (
            OrderedDict()
        )
        self._cache_bytes = 0
        self._max_cache_entries = max_cache_entries
        self._max_cache_bytes = max_cache_bytes

    def analyze(
        self,
        *,
        source_key: tuple[Any, ...],
        request: AnalysisRequest,
        reference: np.ndarray,
        baseline: AnalysisRecord | None,
        target: AnalysisRecord | None,
        sample_rate_hz: float,
        center_frequency_hz: float,
    ) -> dict[str, Any]:
        key = (*source_key, request.fingerprint())
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if not self._gate.acquire(blocking=False):
            raise WebAnalysisError(
                "analysis_busy",
                "another RF analysis is already running",
                status_code=429,
            )
        try:
            cached = self._cache_get(key)
            if cached is not None:
                return cached
            result = analyze_waveforms(
                request=request,
                reference=reference,
                baseline=baseline,
                target=target,
                sample_rate_hz=sample_rate_hz,
                center_frequency_hz=center_frequency_hz,
            )
            self._cache_put(key, result)
            return copy.deepcopy(result)
        finally:
            self._gate.release()

    def _cache_get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        with self._cache_lock:
            item = self._cache.pop(key, None)
            if item is None:
                return None
            self._cache[key] = item
            return copy.deepcopy(item[0])

    def _cache_put(self, key: tuple[Any, ...], result: dict[str, Any]) -> None:
        size = len(
            json.dumps(result, allow_nan=False, separators=(",", ":")).encode("utf-8")
        )
        if size > self._max_cache_bytes:
            return
        with self._cache_lock:
            previous = self._cache.pop(key, None)
            if previous is not None:
                self._cache_bytes -= previous[1]
            self._cache[key] = (copy.deepcopy(result), size)
            self._cache_bytes += size
            while self._cache and (
                len(self._cache) > self._max_cache_entries
                or self._cache_bytes > self._max_cache_bytes
            ):
                _, (_, removed_size) = self._cache.popitem(last=False)
                self._cache_bytes -= removed_size


def analyze_waveforms(
    *,
    request: AnalysisRequest,
    reference: np.ndarray,
    baseline: AnalysisRecord | None,
    target: AnalysisRecord | None,
    sample_rate_hz: float,
    center_frequency_hz: float,
) -> dict[str, Any]:
    """Compute bounded RF results from complete periodic waveforms."""
    x = _signal(reference, "reference")
    sample_rate = _positive_real(sample_rate_hz, "sample_rate_hz")
    center_frequency = _positive_real(center_frequency_hz, "center_frequency_hz")
    baseline = _record(baseline, x.size, "baseline")
    target = _record(target, x.size, "target")
    _validate_bands(request.bands, sample_rate)

    trace_sources = _trace_sources(request.traces, x, baseline, target)
    if not trace_sources:
        trace_sources = [("reference_x", "REFERENCE · X", None, x)]
    bin_width_hz = sample_rate / x.size
    first_frequency_hz = -(x.size // 2) * bin_width_hz
    buckets = _display_buckets(x.size, request.points)
    display_frequency = np.asarray(
        [
            first_frequency_hz + ((start + stop - 1) // 2) * bin_width_hz
            for start, stop in buckets
        ],
        dtype=np.float64,
    )
    if request.frequency_mode == "absolute":
        display_frequency = display_frequency + center_frequency

    traces: list[dict[str, Any]] = []
    band_values: dict[str, dict[str, float]] = {}
    for key, label, iteration, signal in trace_sources:
        normalized_spectrum = np.fft.fftshift(np.fft.fft(signal)) / signal.size
        power = np.abs(normalized_spectrum) ** 2
        compressed = np.asarray(
            [float(np.max(power[start:stop])) for start, stop in buckets],
            dtype=np.float64,
        )
        band_values[key] = {
            band.label: _band_power(
                power,
                first_frequency_hz,
                bin_width_hz,
                sample_rate,
                band,
            )
            for band in request.bands
            if band.enabled
        }
        traces.append(
            {
                "key": key,
                "label": label,
                "iteration": iteration,
                "unit": "dBFS/bin",
                "values_dbfs": _power_db(compressed),
                **_signal_metrics(signal),
            }
        )
        del normalized_spectrum, power, compressed

    bands = _band_payload(request.bands, band_values, [item["key"] for item in traces])
    comparison = _comparison_payload(x, baseline, target)
    stimulus_response = _stimulus_response_payload(
        baseline,
        target,
        amplitude_floor_db=request.amplitude_floor_db,
    )
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "profile": request.to_dict(),
        "sample_count": int(x.size),
        "sample_rate_hz": sample_rate,
        "center_frequency_hz": center_frequency,
        "fft_size": int(x.size),
        "bin_width_hz": bin_width_hz,
        "frequency_mode": request.frequency_mode,
        "frequency_hz": display_frequency.tolist(),
        "trace_count": len(traces),
        "traces": traces,
        "bands": bands,
        "aclr_available": any(
            item["role"] == "adjacent" and item.get("aclr") for item in bands
        ),
        "comparison": comparison,
        "stimulus_response": stimulus_response,
    }


def _measurement_band(value: Any) -> MeasurementBand:
    if not isinstance(value, Mapping):
        raise WebAnalysisError(
            "invalid_measurement_bands",
            "each measurement band must be a JSON object",
        )
    allowed = {
        "label",
        "role",
        "center_offset_hz",
        "integration_bandwidth_hz",
        "enabled",
    }
    unknown = set(value) - allowed
    if unknown:
        raise WebAnalysisError(
            "invalid_measurement_bands",
            f"unsupported measurement-band fields: {sorted(unknown)}",
        )
    label = value.get("label")
    if not isinstance(label, str) or not label.strip():
        raise WebAnalysisError(
            "invalid_measurement_bands", "measurement-band label must not be empty"
        )
    label = label.strip()
    if len(label) > MAX_BAND_LABEL_LENGTH:
        raise WebAnalysisError(
            "invalid_measurement_bands",
            f"measurement-band label exceeds {MAX_BAND_LABEL_LENGTH} characters",
        )
    role = value.get("role")
    if not isinstance(role, str) or role not in _BAND_ROLES:
        raise WebAnalysisError(
            "invalid_measurement_bands",
            "measurement-band role must be main, adjacent, or other",
        )
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise WebAnalysisError(
            "invalid_measurement_bands",
            "measurement-band enabled must be a boolean",
        )
    return MeasurementBand(
        label=label,
        role=role,
        center_offset_hz=_finite_real(
            value.get("center_offset_hz"), "center_offset_hz"
        ),
        integration_bandwidth_hz=_positive_real(
            value.get("integration_bandwidth_hz"), "integration_bandwidth_hz"
        ),
        enabled=enabled,
    )


def _validate_bands(bands: Sequence[MeasurementBand], sample_rate_hz: float) -> None:
    half_rate = sample_rate_hz / 2.0
    enabled_main = 0
    for band in bands:
        if not band.enabled:
            continue
        enabled_main += band.role == "main"
        lower = band.center_offset_hz - band.integration_bandwidth_hz / 2.0
        upper = band.center_offset_hz + band.integration_bandwidth_hz / 2.0
        if lower < -half_rate or upper > half_rate:
            raise WebAnalysisError(
                "measurement_band_out_of_range",
                f"measurement band {band.label!r} exceeds the Nyquist interval",
            )
    if enabled_main > 1:
        raise WebAnalysisError(
            "invalid_measurement_bands",
            "at most one enabled main measurement band is allowed",
        )


def _trace_sources(
    requested: Sequence[str],
    reference: np.ndarray,
    baseline: AnalysisRecord | None,
    target: AnalysisRecord | None,
) -> list[tuple[str, str, int | None, np.ndarray]]:
    values: list[tuple[str, str, int | None, np.ndarray]] = []
    for key in requested:
        if key == "baseline_z" and baseline is not None:
            values.append(
                (
                    key,
                    f"PA OUT · ITER {baseline.iteration}",
                    baseline.iteration,
                    baseline.z,
                )
            )
        elif (
            key == "target_z"
            and target is not None
            and (baseline is None or target.iteration != baseline.iteration)
        ):
            values.append(
                (key, f"PA OUT · ITER {target.iteration}", target.iteration, target.z)
            )
        elif key == "reference_x":
            values.append((key, "REFERENCE · X", None, reference))
        elif key == "target_y" and target is not None:
            values.append(
                (key, f"DPD DRIVE · Y{target.iteration}", target.iteration, target.y)
            )
    return values


def _display_buckets(sample_count: int, points: int) -> tuple[tuple[int, int], ...]:
    if sample_count <= points:
        return tuple((index, index + 1) for index in range(sample_count))
    edges = np.linspace(0, sample_count, points + 1, dtype=np.int64)
    return tuple((int(edges[index]), int(edges[index + 1])) for index in range(points))


def _band_power(
    power: np.ndarray,
    first_frequency_hz: float,
    bin_width_hz: float,
    sample_rate_hz: float,
    band: MeasurementBand,
) -> float:
    lower = band.center_offset_hz - band.integration_bandwidth_hz / 2.0
    upper = band.center_offset_hz + band.integration_bandwidth_hz / 2.0
    first_index = max(
        0,
        math.floor((lower - bin_width_hz / 2.0 - first_frequency_hz) / bin_width_hz)
        + 1,
    )
    stop_index = min(
        power.size,
        math.ceil((upper + bin_width_hz / 2.0 - first_frequency_hz) / bin_width_hz),
    )
    integrated = 0.0
    if first_index < stop_index:
        edge_indices = {first_index, stop_index - 1}
        if stop_index - first_index > 2:
            integrated += float(np.sum(power[first_index + 1 : stop_index - 1]))
        for index in edge_indices:
            center = first_frequency_hz + index * bin_width_hz
            overlap = min(center + bin_width_hz / 2.0, upper) - max(
                center - bin_width_hz / 2.0, lower
            )
            integrated += float(power[index]) * min(
                1.0, max(0.0, overlap / bin_width_hz)
            )
    wrapped_center = first_frequency_hz + sample_rate_hz
    wrapped_overlap = min(wrapped_center + bin_width_hz / 2.0, upper) - max(
        wrapped_center - bin_width_hz / 2.0, lower
    )
    if wrapped_overlap > 0.0:
        integrated += float(power[0]) * min(1.0, wrapped_overlap / bin_width_hz)
    return _scalar_power_db(integrated)


def _band_payload(
    bands: Sequence[MeasurementBand],
    values: Mapping[str, Mapping[str, float]],
    trace_keys: Sequence[str],
) -> list[dict[str, Any]]:
    enabled = [band for band in bands if band.enabled]
    mains = [band for band in enabled if band.role == "main"]
    main = mains[0] if len(mains) == 1 else None
    result: list[dict[str, Any]] = []
    for band in bands:
        trace_payload: dict[str, Any] = {}
        for key in trace_keys:
            power_dbfs = values.get(key, {}).get(band.label)
            entry: dict[str, float | None] = {"power_dbfs": power_dbfs}
            if band.enabled and main is not None and power_dbfs is not None:
                main_power = values.get(key, {}).get(main.label)
                if main_power is not None:
                    entry["relative_power_dbc"] = power_dbfs - main_power
                    if band.role == "adjacent":
                        entry["aclr_db"] = main_power - power_dbfs
            trace_payload[key] = entry
        result.append(
            {
                **band.to_dict(),
                "traces": trace_payload,
                "aclr": band.enabled and band.role == "adjacent" and main is not None,
            }
        )
    return result


def _comparison_payload(
    reference: np.ndarray,
    baseline: AnalysisRecord | None,
    target: AnalysisRecord | None,
) -> dict[str, Any]:
    baseline_payload = _record_metrics(reference, baseline)
    target_payload = _record_metrics(reference, target)
    improvement = None
    if baseline_payload is not None and target_payload is not None:
        improvement = {
            "nmse_db": baseline_payload["nmse_db"] - target_payload["nmse_db"],
            "power_db": _optional_delta(
                target_payload["power_dbm"], baseline_payload["power_dbm"]
            ),
            "drive_papr_db": target_payload["drive_papr_db"]
            - baseline_payload["drive_papr_db"],
        }
    return {
        "baseline": baseline_payload,
        "target": target_payload,
        "improvement": improvement,
    }


def _record_metrics(
    reference: np.ndarray, record: AnalysisRecord | None
) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "iteration": record.iteration,
        "nmse_db": _nmse_db(reference, record.z),
        "power_dbm": record.power_dbm,
        "attenuation_db": record.attenuation_db,
        "drive_papr_db": _papr_db(record.y),
        "output_papr_db": _papr_db(record.z),
    }


def _stimulus_response_payload(
    baseline: AnalysisRecord | None,
    target: AnalysisRecord | None,
    *,
    amplitude_floor_db: float,
) -> dict[str, Any]:
    records = [record for record in (baseline, target) if record is not None]
    if not records:
        return {"available": False, "amplitude_floor_db": amplitude_floor_db}
    maximum = max(_signal_peak(record.y) for record in records)
    if maximum <= 0.0:
        return {"available": False, "amplitude_floor_db": amplitude_floor_db}
    edges = np.linspace(0.0, maximum, AM_BINS + 1)
    sample_count = records[0].y.size
    selected_count = min(sample_count, MAX_STIMULUS_RESPONSE_SAMPLES)
    indices = np.unique(
        np.linspace(0, sample_count - 1, selected_count, dtype=np.int64)
    )
    return {
        "available": True,
        "amplitude_floor_db": amplitude_floor_db,
        "input_sample_count": sample_count,
        "analyzed_sample_count": int(indices.size),
        "sampled": indices.size < sample_count,
        "baseline": _binned_response(baseline, edges, amplitude_floor_db, indices),
        "target": _binned_response(target, edges, amplitude_floor_db, indices),
    }


def _binned_response(
    record: AnalysisRecord | None,
    edges: np.ndarray,
    amplitude_floor_db: float,
    indices: np.ndarray,
) -> dict[str, Any] | None:
    if record is None:
        return None
    selected_y = record.y[indices]
    selected_z = record.z[indices]
    input_amplitude = np.abs(selected_y)
    output_amplitude = np.abs(selected_z)
    threshold = float(np.max(input_amplitude)) * 10.0 ** (amplitude_floor_db / 20.0)
    valid = input_amplitude >= threshold
    bin_index = np.digitize(input_amplitude, edges[1:-1], right=False)
    points: list[dict[str, Any]] = []
    for index in range(edges.size - 1):
        selected = valid & (bin_index == index)
        if not np.any(selected):
            continue
        input_values = input_amplitude[selected]
        output_values = output_amplitude[selected]
        gain_values = 20.0 * np.log10(
            np.maximum(output_values, np.finfo(float).tiny) / input_values
        )
        phase_values = np.angle(selected_z[selected] * np.conj(selected_y[selected]))
        phase_center = float(np.angle(np.mean(np.exp(1j * phase_values))))
        phase_delta = np.angle(np.exp(1j * (phase_values - phase_center)))
        points.append(
            {
                "input_amplitude": float(np.median(input_values)),
                "output_amplitude": float(np.median(output_values)),
                "output_low": float(np.percentile(output_values, 10.0)),
                "output_high": float(np.percentile(output_values, 90.0)),
                "gain_db": float(np.median(gain_values)),
                "gain_low_db": float(np.percentile(gain_values, 10.0)),
                "gain_high_db": float(np.percentile(gain_values, 90.0)),
                "phase_degrees": math.degrees(phase_center),
                "phase_low_degrees": math.degrees(
                    phase_center + float(np.percentile(phase_delta, 10.0))
                ),
                "phase_high_degrees": math.degrees(
                    phase_center + float(np.percentile(phase_delta, 90.0))
                ),
                "sample_count": int(np.count_nonzero(selected)),
            }
        )
    return {"iteration": record.iteration, "points": points}


def _signal_metrics(signal: np.ndarray) -> dict[str, float]:
    peak, rms = _signal_peak_rms(signal)
    return {"rms": rms, "peak": peak, "papr_db": 20.0 * math.log10(peak / rms)}


def _power_db(power: np.ndarray) -> list[float]:
    floor_power = 10.0 ** (SPECTRUM_FLOOR_DBFS / 10.0)
    return (10.0 * np.log10(np.maximum(power, floor_power))).tolist()


def _scalar_power_db(power: float) -> float:
    floor_power = 10.0 ** (SPECTRUM_FLOOR_DBFS / 10.0)
    return 10.0 * math.log10(max(power, floor_power))


def _nmse_db(reference: np.ndarray, measured: np.ndarray) -> float:
    denominator_sum = 0.0
    numerator_sum = 0.0
    for start in range(0, reference.size, _CHUNK_SAMPLES):
        stop = min(reference.size, start + _CHUNK_SAMPLES)
        reference_chunk = reference[start:stop]
        error_chunk = measured[start:stop] - reference_chunk
        denominator_sum += float(np.sum(np.abs(reference_chunk) ** 2))
        numerator_sum += float(np.sum(np.abs(error_chunk) ** 2))
    denominator = denominator_sum / reference.size
    numerator = numerator_sum / reference.size
    if numerator <= 0.0:
        return SPECTRUM_FLOOR_DBFS
    return 10.0 * math.log10(numerator / denominator)


def _papr_db(signal: np.ndarray) -> float:
    peak, rms = _signal_peak_rms(signal)
    return 20.0 * math.log10(peak / rms)


def _signal_peak(signal: np.ndarray) -> float:
    peak = 0.0
    for start in range(0, signal.size, _CHUNK_SAMPLES):
        stop = min(signal.size, start + _CHUNK_SAMPLES)
        peak = max(peak, float(np.max(np.abs(signal[start:stop]))))
    return peak


def _signal_peak_rms(signal: np.ndarray) -> tuple[float, float]:
    peak = 0.0
    power_sum = 0.0
    for start in range(0, signal.size, _CHUNK_SAMPLES):
        stop = min(signal.size, start + _CHUNK_SAMPLES)
        magnitude = np.abs(signal[start:stop])
        peak = max(peak, float(np.max(magnitude)))
        power_sum += float(np.sum(magnitude**2))
    return peak, math.sqrt(power_sum / signal.size)


def _record(
    value: AnalysisRecord | None, sample_count: int, name: str
) -> AnalysisRecord | None:
    if value is None:
        return None
    if not isinstance(value, AnalysisRecord):
        raise TypeError(f"{name} must be an AnalysisRecord")
    iteration = _integer(value.iteration, f"{name}.iteration", minimum=0)
    y = _signal(value.y, f"{name}.y")
    z = _signal(value.z, f"{name}.z")
    if y.size != sample_count or z.size != sample_count:
        raise WebAnalysisError(
            "analysis_shape_mismatch",
            f"{name} waveforms must match the reference length",
        )
    return AnalysisRecord(
        iteration=iteration,
        y=y,
        z=z,
        power_dbm=_optional_finite(value.power_dbm, f"{name}.power_dbm"),
        attenuation_db=_optional_finite(value.attenuation_db, f"{name}.attenuation_db"),
        nmse_db=_optional_finite(value.nmse_db, f"{name}.nmse_db"),
    )


def _signal(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 1 or array.size == 0:
        raise WebAnalysisError(
            "invalid_analysis_waveform", f"{name} must be a non-empty vector"
        )
    finite = True
    for start in range(0, array.size, _CHUNK_SAMPLES):
        stop = min(array.size, start + _CHUNK_SAMPLES)
        chunk = array[start:stop]
        if not np.all(np.isfinite(chunk.real)) or not np.all(np.isfinite(chunk.imag)):
            finite = False
            break
    if not finite:
        raise WebAnalysisError(
            "invalid_analysis_waveform", f"{name} must contain finite samples"
        )
    if _signal_peak_rms(array)[1] <= 0.0:
        raise WebAnalysisError(
            "invalid_analysis_waveform", f"{name} must have non-zero RMS"
        )
    return array


def _optional_iteration(value: Any, name: str) -> int | None:
    return None if value is None else _integer(value, name, minimum=0)


def _integer(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise WebAnalysisError("invalid_analysis_request", f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum or (maximum is not None and normalized > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise WebAnalysisError(
            "invalid_analysis_request",
            f"{name} must be at least {minimum}{suffix}",
        )
    return normalized


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise WebAnalysisError("invalid_analysis_request", f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise WebAnalysisError("invalid_analysis_request", f"{name} must be finite")
    return normalized


def _positive_real(value: Any, name: str) -> float:
    normalized = _finite_real(value, name)
    if normalized <= 0.0:
        raise WebAnalysisError(
            "invalid_analysis_request", f"{name} must be greater than zero"
        )
    return normalized


def _optional_finite(value: Any, name: str) -> float | None:
    return None if value is None else _finite_real(value, name)


def _optional_delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right
