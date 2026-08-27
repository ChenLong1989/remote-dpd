"""Typed configuration and capability contracts for RF devices."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import Any

import numpy as np

from .preprocessing import CaptureBatch

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class _FrozenJsonList(list):
    """JSON-serializable list that rejects in-place mutation."""

    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("JSON array is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


class _FrozenJsonDict(dict):
    """JSON-serializable dictionary that rejects in-place mutation."""

    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("JSON object is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _freeze_json_value(value: object, path: str) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (Integral, np.integer)):
        return int(value)
    if isinstance(value, (Real, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{path} must be finite")
        return result
    if isinstance(value, (list, tuple)):
        return _FrozenJsonList(
            _freeze_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            normalized_key = _non_empty_string(f"{path} key", key)
            result[normalized_key] = _freeze_json_value(
                item, f"{path}.{normalized_key}"
            )
        return _FrozenJsonDict(result)
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _thaw_json_value(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json_value(item) for item in value]
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Common settings shared by simulated and physical RF benches."""

    center_frequency_hz: float = 3.5e9
    sample_rate_hz: float = 983.04e6
    tx_channel: str = "0"
    rx_channel: str = "0"
    trigger: str = "immediate"
    average_segment_count: int = 10
    target_power_dbm: float = -10.0
    safety_power_limit_dbm: float = 0.0
    initial_attenuation_db: float = 30.0
    min_attenuation_db: float = 0.0
    max_attenuation_db: float = 60.0
    settle_seconds: float = 0.1
    max_adjustments: int = 100
    call_timeout_seconds: float = 10.0
    device_options: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        center_frequency_hz = _finite_number(
            "center_frequency_hz", self.center_frequency_hz
        )
        sample_rate_hz = _finite_number("sample_rate_hz", self.sample_rate_hz)
        target_power_dbm = _finite_number("target_power_dbm", self.target_power_dbm)
        safety_power_limit_dbm = _finite_number(
            "safety_power_limit_dbm", self.safety_power_limit_dbm
        )
        initial_attenuation_db = _finite_number(
            "initial_attenuation_db", self.initial_attenuation_db
        )
        min_attenuation_db = _finite_number(
            "min_attenuation_db", self.min_attenuation_db
        )
        max_attenuation_db = _finite_number(
            "max_attenuation_db", self.max_attenuation_db
        )
        settle_seconds = _finite_number("settle_seconds", self.settle_seconds)
        call_timeout_seconds = _finite_number(
            "call_timeout_seconds", self.call_timeout_seconds
        )

        if center_frequency_hz <= 0.0:
            raise ValueError("center_frequency_hz must be greater than zero")
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be greater than zero")
        if target_power_dbm > safety_power_limit_dbm:
            raise ValueError("target_power_dbm must not exceed safety_power_limit_dbm")
        if min_attenuation_db < 0.0:
            raise ValueError("min_attenuation_db must not be negative")
        if min_attenuation_db > max_attenuation_db:
            raise ValueError("min_attenuation_db must not exceed max_attenuation_db")
        if not min_attenuation_db <= initial_attenuation_db <= max_attenuation_db:
            raise ValueError(
                "initial_attenuation_db must be within the configured attenuation range"
            )
        if settle_seconds < 0.0:
            raise ValueError("settle_seconds must not be negative")
        if call_timeout_seconds <= 0.0:
            raise ValueError("call_timeout_seconds must be greater than zero")

        _non_empty_string("tx_channel", self.tx_channel)
        _non_empty_string("rx_channel", self.rx_channel)
        _non_empty_string("trigger", self.trigger)
        average_segment_count = _positive_integer(
            "average_segment_count", self.average_segment_count
        )
        max_adjustments = _positive_integer("max_adjustments", self.max_adjustments)

        if not isinstance(self.device_options, Mapping):
            raise TypeError("device_options must be a mapping")
        normalized_options = _freeze_json_value(self.device_options, "device_options")
        if not isinstance(normalized_options, _FrozenJsonDict):  # pragma: no cover
            raise TypeError("device_options must normalize to a JSON object")

        object.__setattr__(self, "center_frequency_hz", center_frequency_hz)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "target_power_dbm", target_power_dbm)
        object.__setattr__(self, "safety_power_limit_dbm", safety_power_limit_dbm)
        object.__setattr__(self, "initial_attenuation_db", initial_attenuation_db)
        object.__setattr__(self, "min_attenuation_db", min_attenuation_db)
        object.__setattr__(self, "max_attenuation_db", max_attenuation_db)
        object.__setattr__(self, "settle_seconds", settle_seconds)
        object.__setattr__(self, "call_timeout_seconds", call_timeout_seconds)
        object.__setattr__(self, "average_segment_count", average_segment_count)
        object.__setattr__(self, "max_adjustments", max_adjustments)
        object.__setattr__(self, "device_options", normalized_options)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a detached built-in JSON representation of this configuration."""

        return {
            "center_frequency_hz": self.center_frequency_hz,
            "sample_rate_hz": self.sample_rate_hz,
            "tx_channel": self.tx_channel,
            "rx_channel": self.rx_channel,
            "trigger": self.trigger,
            "average_segment_count": self.average_segment_count,
            "target_power_dbm": self.target_power_dbm,
            "safety_power_limit_dbm": self.safety_power_limit_dbm,
            "initial_attenuation_db": self.initial_attenuation_db,
            "min_attenuation_db": self.min_attenuation_db,
            "max_attenuation_db": self.max_attenuation_db,
            "settle_seconds": self.settle_seconds,
            "max_adjustments": self.max_adjustments,
            "call_timeout_seconds": self.call_timeout_seconds,
            "device_options": _thaw_json_value(self.device_options),
        }


class DeviceParameterType(str, Enum):
    """Portable value types supported by dynamic device configuration forms."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass(frozen=True, slots=True)
class DeviceParameterField:
    """Schema for one adapter-specific configuration value."""

    name: str
    value_type: DeviceParameterType
    unit: str | None = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    enum_values: tuple[JsonValue, ...] = ()
    default: JsonValue = None
    required: bool = False
    description: str = ""
    step: float | int | None = None
    items: DeviceParameterField | None = None
    properties: tuple[DeviceParameterField, ...] = ()
    additional_properties: bool = True

    def __post_init__(self) -> None:
        _non_empty_string("name", self.name)
        try:
            value_type = DeviceParameterType(self.value_type)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported device parameter type {self.value_type!r}"
            ) from exc
        object.__setattr__(self, "value_type", value_type)

        if self.unit is not None:
            _non_empty_string("unit", self.unit)
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")

        minimum = self._validate_bound("minimum", self.minimum)
        maximum = self._validate_bound("maximum", self.maximum)
        step = self._validate_bound("step", self.step)
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum must not exceed maximum")
        if step is not None and step <= 0:
            raise ValueError("step must be greater than zero")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "step", step)

        if self.items is not None:
            if self.value_type is not DeviceParameterType.ARRAY:
                raise ValueError("items is only valid for array fields")
            if not isinstance(self.items, DeviceParameterField):
                raise TypeError("items must be a DeviceParameterField")
        if not isinstance(self.properties, tuple):
            raise TypeError("properties must be a tuple")
        if self.properties and self.value_type is not DeviceParameterType.OBJECT:
            raise ValueError("properties is only valid for object fields")
        if not isinstance(self.additional_properties, bool):
            raise TypeError("additional_properties must be a boolean")
        property_names: set[str] = set()
        for item in self.properties:
            if not isinstance(item, DeviceParameterField):
                raise TypeError(
                    "properties must contain DeviceParameterField instances"
                )
            if item.name in property_names:
                raise ValueError(f"duplicate object property {item.name!r}")
            property_names.add(item.name)

        if not isinstance(self.enum_values, tuple):
            raise TypeError("enum_values must be a tuple")
        normalized_enum: list[JsonValue] = []
        for index, value in enumerate(self.enum_values):
            normalized = self._normalize_value(
                value,
                label=f"enum_values[{index}]",
                check_enum=False,
            )
            if any(normalized == previous for previous in normalized_enum):
                raise ValueError("enum_values must not contain duplicates")
            normalized_enum.append(normalized)
        object.__setattr__(self, "enum_values", tuple(normalized_enum))
        if self.default is not None:
            object.__setattr__(
                self,
                "default",
                self._normalize_value(self.default, label="default"),
            )

    def _validate_bound(self, name: str, value: float | None) -> float | int | None:
        if value is None:
            return None
        if self.value_type not in (
            DeviceParameterType.INTEGER,
            DeviceParameterType.NUMBER,
        ):
            raise ValueError(f"{name} is only valid for numeric fields")
        if self.value_type is DeviceParameterType.INTEGER:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer for an integer field")
            return int(value)
        return _finite_number(name, value)

    def validate(self, value: object, *, label: str | None = None) -> None:
        """Validate a value against this field and raise on the first violation."""

        self._normalize_value(value, label=self.name if label is None else label)

    def _normalize_value(
        self,
        value: object,
        *,
        label: str,
        check_enum: bool = True,
    ) -> JsonValue:
        normalized = _freeze_json_value(value, label)
        if self.value_type is DeviceParameterType.STRING:
            if not isinstance(normalized, str):
                raise TypeError(f"{label} must be a string")
        elif self.value_type is DeviceParameterType.INTEGER:
            if isinstance(normalized, bool) or not isinstance(normalized, int):
                raise TypeError(f"{label} must be an integer")
        elif self.value_type is DeviceParameterType.NUMBER:
            if isinstance(normalized, bool) or not isinstance(normalized, (int, float)):
                raise TypeError(f"{label} must be a real number")
        elif self.value_type is DeviceParameterType.BOOLEAN:
            if not isinstance(normalized, bool):
                raise TypeError(f"{label} must be a boolean")
        elif self.value_type is DeviceParameterType.ARRAY:
            if not isinstance(normalized, _FrozenJsonList):
                raise TypeError(f"{label} must be an array")
            if self.items is not None:
                normalized = _FrozenJsonList(
                    self.items._normalize_value(item, label=f"{label}[{index}]")
                    for index, item in enumerate(normalized)
                )
        elif self.value_type is DeviceParameterType.OBJECT:
            if not isinstance(normalized, _FrozenJsonDict):
                raise TypeError(f"{label} must be an object")
            normalized = self._normalize_object(normalized, label)

        if self.minimum is not None and normalized < self.minimum:  # type: ignore[operator]
            raise ValueError(f"{label} must be at least {self.minimum}")
        if self.maximum is not None and normalized > self.maximum:  # type: ignore[operator]
            raise ValueError(f"{label} must be at most {self.maximum}")
        if self.step is not None:
            self._validate_step(normalized, label)
        if check_enum and self.enum_values and normalized not in self.enum_values:
            raise ValueError(f"{label} must be one of {self.enum_values!r}")
        return normalized

    def _normalize_object(
        self,
        value: _FrozenJsonDict,
        label: str,
    ) -> _FrozenJsonDict:
        if not self.properties:
            if value and not self.additional_properties:
                raise ValueError(
                    f"{label} contains unknown properties: {sorted(value)!r}"
                )
            return value
        by_name = {item.name: item for item in self.properties}
        unknown = set(value) - set(by_name)
        if unknown and not self.additional_properties:
            raise ValueError(
                f"{label} contains unknown properties: {sorted(unknown)!r}"
            )

        normalized: dict[str, JsonValue] = {}
        for item in self.properties:
            property_label = f"{label}.{item.name}"
            if item.name in value:
                normalized[item.name] = item._normalize_value(
                    value[item.name],
                    label=property_label,
                )
            elif item.default is not None:
                normalized[item.name] = item.default
            elif item.required:
                raise ValueError(f"missing required property {property_label!r}")
        if self.additional_properties:
            for name, item in value.items():
                if name not in by_name:
                    normalized[name] = item
        return _FrozenJsonDict(normalized)

    def _validate_step(self, value: JsonValue, label: str) -> None:
        if self.value_type is DeviceParameterType.INTEGER:
            base = int(self.minimum or 0)
            if (int(value) - base) % int(self.step) != 0:  # type: ignore[arg-type]
                raise ValueError(f"{label} must follow step {self.step} from {base}")
            return
        base = float(self.minimum or 0.0)
        quotient = (float(value) - base) / float(self.step)  # type: ignore[arg-type]
        if not math.isclose(quotient, round(quotient), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{label} must follow step {self.step} from {base}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for API and UI consumers."""

        return {
            "name": self.name,
            "type": self.value_type.value,
            "unit": self.unit,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "enum": [_thaw_json_value(item) for item in self.enum_values],
            "default": _thaw_json_value(self.default),
            "required": self.required,
            "description": self.description,
            "items": None if self.items is None else self.items.to_dict(),
            "properties": [item.to_dict() for item in self.properties],
            "additional_properties": self.additional_properties,
        }


@dataclass(frozen=True, slots=True)
class DeviceParameterSchema:
    """Versioned schema published by an RF bench adapter."""

    device_type: str
    schema_version: int
    fields: tuple[DeviceParameterField, ...] = ()

    def __post_init__(self) -> None:
        _non_empty_string("device_type", self.device_type)
        schema_version = _positive_integer("schema_version", self.schema_version)
        object.__setattr__(self, "schema_version", schema_version)
        if not isinstance(self.fields, tuple):
            raise TypeError("fields must be a tuple")
        names: set[str] = set()
        for item in self.fields:
            if not isinstance(item, DeviceParameterField):
                raise TypeError("fields must contain DeviceParameterField instances")
            if item.name in names:
                raise ValueError(f"duplicate device parameter field {item.name!r}")
            names.add(item.name)

    def validate_options(
        self, options: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Validate adapter options and apply defaults to omitted optional fields."""

        if not isinstance(options, Mapping):
            raise TypeError("options must be a mapping")
        normalized_options = _freeze_json_value(options, "options")
        if not isinstance(normalized_options, _FrozenJsonDict):  # pragma: no cover
            raise TypeError("options must normalize to a JSON object")
        by_name = {item.name: item for item in self.fields}
        unknown = set(normalized_options) - set(by_name)
        if unknown:
            raise ValueError(f"unknown device options: {sorted(unknown)!r}")

        validated: dict[str, JsonValue] = {}
        for item in self.fields:
            if item.name in normalized_options:
                value = item._normalize_value(
                    normalized_options[item.name],
                    label=item.name,
                )
                validated[item.name] = _thaw_json_value(value)
            elif item.default is not None:
                validated[item.name] = _thaw_json_value(item.default)
            elif item.required:
                raise ValueError(f"missing required device option {item.name!r}")
        return validated

    def to_dict(self) -> dict[str, Any]:
        """Return the schema in a transport-neutral JSON-compatible form."""

        return {
            "device_type": self.device_type,
            "schema_version": self.schema_version,
            "fields": [item.to_dict() for item in self.fields],
        }


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    """Request for one receiver batch containing complete waveform periods."""

    segment_length: int
    segment_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "segment_length",
            _positive_integer("segment_length", self.segment_length),
        )
        object.__setattr__(
            self,
            "segment_count",
            _positive_integer("segment_count", self.segment_count),
        )

    @property
    def sample_count(self) -> int:
        return self.segment_length * self.segment_count


class DeviceCapability(ABC):
    """Lifecycle shared by independently connectable device capabilities."""

    @abstractmethod
    def connect(self, timeout_seconds: float) -> None:
        """Connect within the supplied timeout."""

    @abstractmethod
    def configure(self, config: DeviceConfig, timeout_seconds: float) -> None:
        """Apply common and adapter-specific settings."""

    @abstractmethod
    def disconnect(self, timeout_seconds: float) -> None:
        """Release the device connection within the supplied timeout."""


class Transmitter(DeviceCapability, ABC):
    """Capability for cyclic waveform transmission and TX attenuation."""

    @abstractmethod
    def upload_waveform(self, waveform: np.ndarray, timeout_seconds: float) -> None:
        """Upload one finite cyclic IQ waveform without modifying its samples."""

    @abstractmethod
    def start_transmission(self, timeout_seconds: float) -> None:
        """Start cyclic transmission of the uploaded waveform."""

    @abstractmethod
    def stop_transmission(self, timeout_seconds: float) -> None:
        """Stop RF transmission safely."""

    @abstractmethod
    def get_attenuation_db(self, timeout_seconds: float) -> float:
        """Return the current adjustable TX attenuation in dB."""

    @abstractmethod
    def set_attenuation_db(self, attenuation_db: float, timeout_seconds: float) -> None:
        """Set adjustable TX attenuation in dB."""


class Receiver(DeviceCapability, ABC):
    """Capability for acquiring complete batches of periodic IQ feedback."""

    @property
    @abstractmethod
    def max_capture_samples(self) -> int:
        """Return the largest sample count supported by one capture call."""

    @abstractmethod
    def capture(self, request: CaptureRequest, timeout_seconds: float) -> CaptureBatch:
        """Capture exactly the requested complete segments as one batch."""


class PowerSensor(DeviceCapability, ABC):
    """Capability for calibrated physical output-power measurements."""

    @abstractmethod
    def measure_power_dbm(self, timeout_seconds: float) -> float:
        """Return one calibrated output-power reading in dBm."""


class RFBench(ABC):
    """Aggregate capability boundary for integrated or split RF equipment."""

    @property
    @abstractmethod
    def transmitter(self) -> Transmitter:
        """Return the configured transmitter capability."""

    @property
    @abstractmethod
    def receiver(self) -> Receiver:
        """Return the configured receiver capability."""

    @property
    @abstractmethod
    def power_sensor(self) -> PowerSensor:
        """Return the configured calibrated power-sensor capability."""

    @property
    @abstractmethod
    def parameter_schema(self) -> DeviceParameterSchema:
        """Return the adapter-specific dynamic parameter schema."""

    @abstractmethod
    def connect(self, timeout_seconds: float) -> None:
        """Connect all unique underlying instruments."""

    @abstractmethod
    def configure(self, config: DeviceConfig, timeout_seconds: float) -> None:
        """Validate and apply one common configuration atomically."""

    @abstractmethod
    def safe_shutdown(self, timeout_seconds: float) -> None:
        """Stop transmission and place all instruments in a safe state."""

    @abstractmethod
    def disconnect(self, timeout_seconds: float) -> None:
        """Release all unique underlying instrument connections."""
