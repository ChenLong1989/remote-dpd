"""Initial output-power tuning and per-iteration safety monitoring."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real

from .device import DeviceConfig, PowerSensor, Transmitter

POWER_TOLERANCE_DB = 0.2
COARSE_GAP_THRESHOLD_DB = 1.0
COARSE_ATTENUATION_STEP_DB = 1.0
FINE_ATTENUATION_STEP_DB = 0.1
_COMPARISON_ABS_TOLERANCE_DB = 1e-12

CancelRequested = Callable[[], bool]
SleepFunction = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class PowerAdjustment:
    """One measured point in an initial power-tuning trace."""

    attenuation_db: float
    power_dbm: float
    gap_db: float


@dataclass(frozen=True, slots=True)
class PowerControlResult:
    """The safe attenuation selected by initial output-power tuning."""

    attenuation_db: float
    power_dbm: float
    gap_db: float
    trace: tuple[PowerAdjustment, ...]


class PowerControlError(RuntimeError):
    """A power measurement or attenuation decision failed safely."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        trace: tuple[PowerAdjustment, ...] = (),
        last_safe_attenuation_db: float | None = None,
        last_safe_power_dbm: float | None = None,
        measured_power_dbm: float | None = None,
        limit_power_dbm: float | None = None,
    ) -> None:
        self.code = code
        self.trace = trace
        self.last_safe_attenuation_db = last_safe_attenuation_db
        self.last_safe_power_dbm = last_safe_power_dbm
        self.measured_power_dbm = measured_power_dbm
        self.limit_power_dbm = limit_power_dbm
        super().__init__(message)


class PowerControlCancelled(PowerControlError):
    """Power control was cancelled at a measurement-safe boundary."""

    def __init__(
        self,
        *,
        trace: tuple[PowerAdjustment, ...] = (),
        last_safe_attenuation_db: float | None = None,
        last_safe_power_dbm: float | None = None,
    ) -> None:
        super().__init__(
            "cancelled",
            "power control was cancelled",
            trace=trace,
            last_safe_attenuation_db=last_safe_attenuation_db,
            last_safe_power_dbm=last_safe_power_dbm,
        )


class PowerController:
    """Tune TX attenuation once and monitor later physical power readings."""

    def __init__(self, sleep_fn: SleepFunction = time.sleep) -> None:
        if not callable(sleep_fn):
            raise TypeError("sleep_fn must be callable")
        self._sleep = sleep_fn

    def tune(
        self,
        transmitter: Transmitter,
        power_sensor: PowerSensor,
        config: DeviceConfig,
        cancel_requested: CancelRequested | None = None,
    ) -> PowerControlResult:
        """Tune an already-running transmitter from below the power target.

        The caller must upload the reference waveform and start cyclic transmission
        before invoking this method. The selected attenuation remains unchanged on
        success. Device shutdown belongs to the enclosing closed-loop controller.
        """

        self._validate_inputs(config, cancel_requested)
        trace: list[PowerAdjustment] = []
        last_safe_attenuation_db: float | None = None
        last_safe_power_dbm: float | None = None

        self._raise_if_cancelled(
            cancel_requested,
            trace,
            last_safe_attenuation_db,
            last_safe_power_dbm,
        )

        attenuation_db = config.initial_attenuation_db
        transmitter.set_attenuation_db(
            attenuation_db,
            timeout_seconds=config.call_timeout_seconds,
        )
        self._sleep(config.settle_seconds)

        power_dbm = power_sensor.measure_power_dbm(
            timeout_seconds=config.call_timeout_seconds
        )
        adjustment = self._record_measurement_or_restore(
            attenuation_db,
            power_dbm,
            transmitter,
            config,
            trace,
            last_safe_attenuation_db,
            last_safe_power_dbm,
        )
        trace.append(adjustment)
        self._validate_tuning_measurement(
            adjustment,
            transmitter,
            config,
            trace,
            last_safe_attenuation_db,
            last_safe_power_dbm,
        )
        last_safe_attenuation_db = attenuation_db
        last_safe_power_dbm = adjustment.power_dbm

        self._raise_if_cancelled(
            cancel_requested,
            trace,
            last_safe_attenuation_db,
            last_safe_power_dbm,
        )
        if _within_power_tolerance(adjustment.gap_db):
            return self._result(adjustment, trace)

        for _ in range(config.max_adjustments):
            step_db = (
                COARSE_ATTENUATION_STEP_DB
                if adjustment.gap_db > COARSE_GAP_THRESHOLD_DB
                else FINE_ATTENUATION_STEP_DB
            )
            next_attenuation_db = round(attenuation_db - step_db, 12)
            if next_attenuation_db < config.min_attenuation_db:
                raise self._error(
                    "attenuation_limit_reached",
                    "the next attenuation step would cross the configured minimum",
                    trace,
                    last_safe_attenuation_db,
                    last_safe_power_dbm,
                )

            transmitter.set_attenuation_db(
                next_attenuation_db,
                timeout_seconds=config.call_timeout_seconds,
            )
            attenuation_db = next_attenuation_db
            self._sleep(config.settle_seconds)

            power_dbm = power_sensor.measure_power_dbm(
                timeout_seconds=config.call_timeout_seconds
            )
            adjustment = self._record_measurement_or_restore(
                attenuation_db,
                power_dbm,
                transmitter,
                config,
                trace,
                last_safe_attenuation_db,
                last_safe_power_dbm,
            )
            trace.append(adjustment)
            self._validate_tuning_measurement(
                adjustment,
                transmitter,
                config,
                trace,
                last_safe_attenuation_db,
                last_safe_power_dbm,
            )
            last_safe_attenuation_db = attenuation_db
            last_safe_power_dbm = adjustment.power_dbm

            self._raise_if_cancelled(
                cancel_requested,
                trace,
                last_safe_attenuation_db,
                last_safe_power_dbm,
            )
            if _within_power_tolerance(adjustment.gap_db):
                return self._result(adjustment, trace)

        raise self._error(
            "max_adjustments_reached",
            "power target was not reached within max_adjustments",
            trace,
            last_safe_attenuation_db,
            last_safe_power_dbm,
        )

    def monitor(
        self,
        power_sensor: PowerSensor,
        config: DeviceConfig,
        cancel_requested: CancelRequested | None = None,
    ) -> float:
        """Measure physical power without changing the locked TX attenuation."""

        return self.measure_safe(power_sensor, config, cancel_requested)

    def measure_safe(
        self,
        power_sensor: PowerSensor,
        config: DeviceConfig,
        cancel_requested: CancelRequested | None = None,
    ) -> float:
        """Return a finite reading at or below the absolute safety limit."""

        self._validate_inputs(config, cancel_requested)
        self._raise_if_cancelled(cancel_requested, [], None, None)
        raw_power_dbm = power_sensor.measure_power_dbm(
            timeout_seconds=config.call_timeout_seconds
        )
        power_dbm = _normalize_power_reading(raw_power_dbm)
        if power_dbm > config.safety_power_limit_dbm:
            raise PowerControlError(
                "safety_limit_exceeded",
                f"measured power {power_dbm} dBm exceeded the absolute safety "
                f"limit {config.safety_power_limit_dbm} dBm",
                measured_power_dbm=power_dbm,
                limit_power_dbm=config.safety_power_limit_dbm,
            )
        self._raise_if_cancelled(cancel_requested, [], None, None)
        return float(power_dbm)

    @staticmethod
    def _validate_inputs(
        config: DeviceConfig,
        cancel_requested: CancelRequested | None,
    ) -> None:
        if not isinstance(config, DeviceConfig):
            raise TypeError("config must be a DeviceConfig")
        if cancel_requested is not None and not callable(cancel_requested):
            raise TypeError("cancel_requested must be callable or None")

    @staticmethod
    def _record_measurement(
        attenuation_db: float,
        power_dbm: object,
        config: DeviceConfig,
    ) -> PowerAdjustment:
        normalized_power_dbm = _normalize_power_reading(power_dbm)
        return PowerAdjustment(
            attenuation_db=float(attenuation_db),
            power_dbm=normalized_power_dbm,
            gap_db=config.target_power_dbm - normalized_power_dbm,
        )

    def _record_measurement_or_restore(
        self,
        attenuation_db: float,
        power_dbm: object,
        transmitter: Transmitter,
        config: DeviceConfig,
        trace: list[PowerAdjustment],
        last_safe_attenuation_db: float | None,
        last_safe_power_dbm: float | None,
    ) -> PowerAdjustment:
        try:
            return self._record_measurement(attenuation_db, power_dbm, config)
        except PowerControlError as exc:
            self._restore_last_safe(
                transmitter,
                config,
                last_safe_attenuation_db,
            )
            raise self._error(
                exc.code,
                str(exc),
                trace,
                last_safe_attenuation_db,
                last_safe_power_dbm,
            ) from exc

    def _validate_tuning_measurement(
        self,
        adjustment: PowerAdjustment,
        transmitter: Transmitter,
        config: DeviceConfig,
        trace: list[PowerAdjustment],
        last_safe_attenuation_db: float | None,
        last_safe_power_dbm: float | None,
    ) -> None:
        power_dbm = adjustment.power_dbm
        if power_dbm > config.safety_power_limit_dbm:
            self._restore_last_safe(
                transmitter,
                config,
                last_safe_attenuation_db,
            )
            raise self._error(
                "safety_limit_exceeded",
                f"measured power {power_dbm} dBm exceeded the absolute safety "
                f"limit {config.safety_power_limit_dbm} dBm",
                trace,
                last_safe_attenuation_db,
                last_safe_power_dbm,
                measured_power_dbm=power_dbm,
                limit_power_dbm=config.safety_power_limit_dbm,
            )
        if power_dbm > config.target_power_dbm:
            self._restore_last_safe(
                transmitter,
                config,
                last_safe_attenuation_db,
            )
            raise self._error(
                "target_overshoot",
                f"measured power {power_dbm} dBm exceeded the target "
                f"{config.target_power_dbm} dBm",
                trace,
                last_safe_attenuation_db,
                last_safe_power_dbm,
                measured_power_dbm=power_dbm,
                limit_power_dbm=config.target_power_dbm,
            )

    @staticmethod
    def _restore_last_safe(
        transmitter: Transmitter,
        config: DeviceConfig,
        last_safe_attenuation_db: float | None,
    ) -> None:
        if last_safe_attenuation_db is None:
            return
        transmitter.set_attenuation_db(
            last_safe_attenuation_db,
            timeout_seconds=config.call_timeout_seconds,
        )

    @staticmethod
    def _raise_if_cancelled(
        cancel_requested: CancelRequested | None,
        trace: list[PowerAdjustment],
        last_safe_attenuation_db: float | None,
        last_safe_power_dbm: float | None,
    ) -> None:
        if cancel_requested is not None and cancel_requested():
            raise PowerControlCancelled(
                trace=tuple(trace),
                last_safe_attenuation_db=last_safe_attenuation_db,
                last_safe_power_dbm=last_safe_power_dbm,
            )

    @staticmethod
    def _result(
        adjustment: PowerAdjustment,
        trace: list[PowerAdjustment],
    ) -> PowerControlResult:
        return PowerControlResult(
            attenuation_db=adjustment.attenuation_db,
            power_dbm=adjustment.power_dbm,
            gap_db=adjustment.gap_db,
            trace=tuple(trace),
        )

    @staticmethod
    def _error(
        code: str,
        message: str,
        trace: list[PowerAdjustment],
        last_safe_attenuation_db: float | None,
        last_safe_power_dbm: float | None,
        *,
        measured_power_dbm: float | None = None,
        limit_power_dbm: float | None = None,
    ) -> PowerControlError:
        return PowerControlError(
            code,
            message,
            trace=tuple(trace),
            last_safe_attenuation_db=last_safe_attenuation_db,
            last_safe_power_dbm=last_safe_power_dbm,
            measured_power_dbm=measured_power_dbm,
            limit_power_dbm=limit_power_dbm,
        )


def _normalize_power_reading(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PowerControlError(
            "invalid_power",
            "power sensor must return a real scalar",
        )
    result = float(value)
    if not math.isfinite(result):
        raise PowerControlError(
            "non_finite_power",
            "power sensor returned a non-finite value",
        )
    return result


def _within_power_tolerance(gap_db: float) -> bool:
    return gap_db <= POWER_TOLERANCE_DB + _COMPARISON_ABS_TOLERANCE_DB
