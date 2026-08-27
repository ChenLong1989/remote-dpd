import inspect
import json
import unittest

import numpy as np

from remote_dpd.device import (
    CaptureRequest,
    DeviceCapability,
    DeviceConfig,
    DeviceParameterField,
    DeviceParameterSchema,
    DeviceParameterType,
    PowerSensor,
    Receiver,
    RFBench,
    Transmitter,
)
from remote_dpd.preprocessing import CaptureBatch


class DeviceConfigTests(unittest.TestCase):
    def test_accepts_and_normalizes_valid_common_configuration(self):
        source_options = {
            "clock": "external",
            "seed": np.int64(7),
            "pa": {"coefficients": [np.float32(1.0), 0.1]},
        }
        config = DeviceConfig(
            center_frequency_hz=3_500_000_000,
            sample_rate_hz=245_760_000,
            tx_channel="tx0",
            rx_channel="rx1",
            trigger="pxi0",
            average_segment_count=8,
            target_power_dbm=-5,
            safety_power_limit_dbm=-2,
            initial_attenuation_db=25,
            min_attenuation_db=1,
            max_attenuation_db=40,
            settle_seconds=0,
            max_adjustments=50,
            call_timeout_seconds=3,
            device_options=source_options,
        )

        self.assertEqual(config.center_frequency_hz, 3.5e9)
        self.assertEqual(config.average_segment_count, 8)
        self.assertEqual(config.device_options["clock"], "external")
        self.assertIs(type(config.device_options["seed"]), int)
        self.assertIs(type(config.device_options["pa"]["coefficients"][0]), float)
        source_options["pa"]["coefficients"][0] = 99.0
        self.assertEqual(config.device_options["pa"]["coefficients"][0], 1.0)
        with self.assertRaises(TypeError):
            config.device_options["new"] = True
        with self.assertRaises(TypeError):
            config.device_options["pa"]["coefficients"].append(3.0)

        serialized = config.to_dict()
        json.dumps(serialized, allow_nan=False)
        json.dumps(config.device_options, allow_nan=False)
        serialized["device_options"]["pa"]["coefficients"][0] = 5.0
        self.assertEqual(config.device_options["pa"]["coefficients"][0], 1.0)

    def test_rejects_invalid_numeric_and_relational_values(self):
        invalid_cases = (
            ({"center_frequency_hz": 0}, ValueError),
            ({"sample_rate_hz": float("nan")}, ValueError),
            ({"average_segment_count": True}, TypeError),
            ({"average_segment_count": 0}, ValueError),
            ({"target_power_dbm": 1, "safety_power_limit_dbm": 0}, ValueError),
            ({"min_attenuation_db": -0.1}, ValueError),
            ({"min_attenuation_db": 20, "max_attenuation_db": 10}, ValueError),
            (
                {
                    "min_attenuation_db": 0,
                    "initial_attenuation_db": 61,
                    "max_attenuation_db": 60,
                },
                ValueError,
            ),
            ({"settle_seconds": -0.01}, ValueError),
            ({"max_adjustments": 1.5}, TypeError),
            ({"call_timeout_seconds": 0}, ValueError),
        )
        for changes, error_type in invalid_cases:
            with self.subTest(changes=changes), self.assertRaises(error_type):
                DeviceConfig(**changes)

    def test_rejects_invalid_identifiers_and_device_options(self):
        with self.assertRaises(ValueError):
            DeviceConfig(tx_channel="  ")
        with self.assertRaises(TypeError):
            DeviceConfig(device_options={"bad": object()})
        with self.assertRaises(ValueError):
            DeviceConfig(device_options={"bad": float("inf")})


class DeviceParameterSchemaTests(unittest.TestCase):
    def test_validates_ranges_enums_required_values_and_defaults(self):
        schema = DeviceParameterSchema(
            device_type="simulated",
            schema_version=1,
            fields=(
                DeviceParameterField(
                    name="noise_dbfs",
                    value_type=DeviceParameterType.NUMBER,
                    unit="dBFS",
                    minimum=-160.0,
                    maximum=0.0,
                    default=-80.0,
                ),
                DeviceParameterField(
                    name="clock_source",
                    value_type=DeviceParameterType.STRING,
                    enum_values=("internal", "external"),
                    required=True,
                ),
                DeviceParameterField(
                    name="seed",
                    value_type=DeviceParameterType.INTEGER,
                    minimum=0,
                    default=7,
                ),
            ),
        )

        result = schema.validate_options(
            {"clock_source": "external", "noise_dbfs": -70}
        )

        self.assertEqual(
            result,
            {"noise_dbfs": -70, "clock_source": "external", "seed": 7},
        )
        self.assertEqual(schema.to_dict()["fields"][0]["unit"], "dBFS")
        self.assertEqual(schema.to_dict()["schema_version"], 1)
        json.dumps(schema.to_dict(), allow_nan=False)

    def test_nested_schema_validates_memory_polynomial_pa_rows(self):
        coefficient_row = DeviceParameterField(
            name="coefficient",
            value_type=DeviceParameterType.OBJECT,
            properties=(
                DeviceParameterField(
                    "p",
                    DeviceParameterType.INTEGER,
                    minimum=1,
                    step=2,
                    required=True,
                ),
                DeviceParameterField(
                    "m",
                    DeviceParameterType.INTEGER,
                    minimum=0,
                    required=True,
                ),
                DeviceParameterField(
                    "real",
                    DeviceParameterType.NUMBER,
                    required=True,
                ),
                DeviceParameterField(
                    "imag",
                    DeviceParameterType.NUMBER,
                    required=True,
                ),
            ),
            additional_properties=False,
        )
        default_rows = [{"p": 1, "m": 0, "real": 1.0, "imag": 0.0}]
        coefficients = DeviceParameterField(
            name="pa_coefficients",
            value_type=DeviceParameterType.ARRAY,
            items=coefficient_row,
            default=default_rows,
        )
        schema = DeviceParameterSchema("simulated", 1, (coefficients,))

        default_rows[0]["real"] = 99.0
        result = schema.validate_options(
            {"pa_coefficients": [{"p": np.int64(3), "m": 2, "real": 0.2, "imag": -0.1}]}
        )

        self.assertEqual(
            result,
            {"pa_coefficients": [{"p": 3, "m": 2, "real": 0.2, "imag": -0.1}]},
        )
        self.assertIs(type(result["pa_coefficients"][0]["p"]), int)
        self.assertEqual(coefficients.default[0]["real"], 1.0)
        with self.assertRaises(TypeError):
            coefficients.default[0]["real"] = 2.0
        json.dumps(schema.to_dict(), allow_nan=False)
        json.dumps(result, allow_nan=False)

        invalid_rows = (
            ({"p": 2, "m": 0, "real": 1.0, "imag": 0.0}, "step"),
            ({"p": 1, "m": -1, "real": 1.0, "imag": 0.0}, "at least"),
            ({"p": 1, "m": 0, "real": 1.0}, "missing required property"),
            (
                {"p": 1, "m": 0, "real": 1.0, "imag": 0.0, "extra": 1},
                "unknown properties",
            ),
        )
        for row, message in invalid_rows:
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, message):
                schema.validate_options({"pa_coefficients": [row]})

    def test_nested_enum_and_default_values_are_detached_and_immutable(self):
        enum_source = {"clock": ["internal"]}
        default_source = {"clock": ["internal"]}
        field = DeviceParameterField(
            name="profile",
            value_type=DeviceParameterType.OBJECT,
            enum_values=(enum_source,),
            default=default_source,
        )

        enum_source["clock"].append("changed")
        default_source["clock"].append("changed")

        self.assertEqual(field.enum_values[0]["clock"], ["internal"])
        self.assertEqual(field.default["clock"], ["internal"])
        with self.assertRaises(TypeError):
            field.enum_values[0]["clock"].append("external")
        with self.assertRaises(TypeError):
            field.default["clock"] = ["external"]
        json.dumps(field.enum_values, allow_nan=False)
        json.dumps(field.to_dict(), allow_nan=False)

    def test_rejects_invalid_schema_definitions(self):
        with self.assertRaises(ValueError):
            DeviceParameterField(
                name="gain",
                value_type=DeviceParameterType.NUMBER,
                minimum=2,
                maximum=1,
            )
        with self.assertRaises(TypeError):
            DeviceParameterField(
                name="count",
                value_type=DeviceParameterType.INTEGER,
                default=True,
            )
        duplicate = DeviceParameterField("mode", DeviceParameterType.STRING)
        with self.assertRaises(ValueError):
            DeviceParameterSchema("simulated", 1, (duplicate, duplicate))
        empty_object = DeviceParameterField(
            "empty",
            DeviceParameterType.OBJECT,
            additional_properties=False,
        )
        with self.assertRaisesRegex(ValueError, "unknown properties"):
            empty_object.validate({"unexpected": 1})

    def test_rejects_unknown_missing_and_invalid_options(self):
        schema = DeviceParameterSchema(
            "simulated",
            1,
            (
                DeviceParameterField(
                    "mode",
                    DeviceParameterType.STRING,
                    enum_values=("linear", "nonlinear"),
                    required=True,
                ),
                DeviceParameterField(
                    "depth",
                    DeviceParameterType.INTEGER,
                    minimum=1,
                    maximum=10,
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "missing required"):
            schema.validate_options({})
        with self.assertRaisesRegex(ValueError, "unknown device options"):
            schema.validate_options({"mode": "linear", "extra": 1})
        with self.assertRaisesRegex(ValueError, "must be one of"):
            schema.validate_options({"mode": "invalid"})
        with self.assertRaisesRegex(ValueError, "at most"):
            schema.validate_options({"mode": "linear", "depth": 11})


class CaptureRequestTests(unittest.TestCase):
    def test_reports_total_sample_count(self):
        request = CaptureRequest(segment_length=4096, segment_count=5)
        self.assertEqual(request.sample_count, 20_480)

    def test_rejects_non_positive_or_non_integer_dimensions(self):
        with self.assertRaises(ValueError):
            CaptureRequest(segment_length=0, segment_count=1)
        with self.assertRaises(TypeError):
            CaptureRequest(segment_length=10, segment_count=1.5)


class _IntegratedInstrument(Transmitter, Receiver, PowerSensor):
    def __init__(self) -> None:
        self.waveform = np.empty(0, dtype=np.complex128)
        self.attenuation_db = 0.0

    def connect(self, timeout_seconds: float) -> None:
        pass

    def configure(self, config: DeviceConfig, timeout_seconds: float) -> None:
        pass

    def disconnect(self, timeout_seconds: float) -> None:
        pass

    def upload_waveform(self, waveform: np.ndarray, timeout_seconds: float) -> None:
        self.waveform = waveform.copy()

    def start_transmission(self, timeout_seconds: float) -> None:
        pass

    def stop_transmission(self, timeout_seconds: float) -> None:
        pass

    def get_attenuation_db(self, timeout_seconds: float) -> float:
        return self.attenuation_db

    def set_attenuation_db(
        self,
        attenuation_db: float,
        timeout_seconds: float,
    ) -> None:
        self.attenuation_db = attenuation_db

    @property
    def max_capture_samples(self) -> int:
        return 1_000_000

    def capture(self, request: CaptureRequest, timeout_seconds: float) -> CaptureBatch:
        return CaptureBatch(
            iq=np.zeros(request.sample_count, dtype=np.complex128),
            segment_length=request.segment_length,
            segment_count=request.segment_count,
            sample_rate_hz=1.0,
        )

    def measure_power_dbm(self, timeout_seconds: float) -> float:
        return -10.0


class _IntegratedBench(RFBench):
    def __init__(self) -> None:
        self.instrument = _IntegratedInstrument()

    @property
    def transmitter(self) -> Transmitter:
        return self.instrument

    @property
    def receiver(self) -> Receiver:
        return self.instrument

    @property
    def power_sensor(self) -> PowerSensor:
        return self.instrument

    @property
    def parameter_schema(self) -> DeviceParameterSchema:
        return DeviceParameterSchema("integrated", 1)

    def connect(self, timeout_seconds: float) -> None:
        self.instrument.connect(timeout_seconds)

    def configure(self, config: DeviceConfig, timeout_seconds: float) -> None:
        self.instrument.configure(config, timeout_seconds)

    def safe_shutdown(self, timeout_seconds: float) -> None:
        self.instrument.stop_transmission(timeout_seconds)

    def disconnect(self, timeout_seconds: float) -> None:
        self.instrument.disconnect(timeout_seconds)


class DeviceCapabilityTests(unittest.TestCase):
    def test_one_integrated_instrument_can_supply_all_bench_capabilities(self):
        bench = _IntegratedBench()

        self.assertIs(bench.transmitter, bench.receiver)
        self.assertIs(bench.receiver, bench.power_sensor)
        self.assertEqual(bench.receiver.max_capture_samples, 1_000_000)
        self.assertEqual(bench.parameter_schema.device_type, "integrated")
        capture = bench.receiver.capture(CaptureRequest(16, 2), timeout_seconds=1.0)
        self.assertIsInstance(capture, CaptureBatch)
        self.assertEqual(capture.segments.shape, (2, 16))

    def test_every_potentially_blocking_capability_method_requires_timeout(self):
        methods = (
            DeviceCapability.configure,
            Transmitter.upload_waveform,
            Transmitter.start_transmission,
            Transmitter.stop_transmission,
            Transmitter.get_attenuation_db,
            Transmitter.set_attenuation_db,
            Receiver.capture,
            PowerSensor.measure_power_dbm,
            RFBench.configure,
        )
        for method in methods:
            with self.subTest(method=method.__qualname__):
                self.assertIn("timeout_seconds", inspect.signature(method).parameters)


if __name__ == "__main__":
    unittest.main()
