import math
import unittest

import numpy as np

from remote_dpd.device import DeviceConfig
from remote_dpd.power_control import (
    PowerControlCancelled,
    PowerControlError,
    PowerController,
)


class _FakeTransmitter:
    def __init__(self) -> None:
        self.set_calls: list[tuple[float, float]] = []

    def set_attenuation_db(
        self,
        attenuation_db: float,
        timeout_seconds: float,
    ) -> None:
        self.set_calls.append((attenuation_db, timeout_seconds))


class _FakePowerSensor:
    def __init__(self, readings: list[float]) -> None:
        self._readings = iter(readings)
        self.timeouts: list[float] = []

    def measure_power_dbm(self, timeout_seconds: float) -> float:
        self.timeouts.append(timeout_seconds)
        return next(self._readings)


def _config(**changes: object) -> DeviceConfig:
    values = {
        "target_power_dbm": -10.0,
        "safety_power_limit_dbm": 0.0,
        "initial_attenuation_db": 30.0,
        "min_attenuation_db": 0.0,
        "max_attenuation_db": 60.0,
        "settle_seconds": 0.25,
        "max_adjustments": 5,
        "call_timeout_seconds": 3.0,
    }
    values.update(changes)
    return DeviceConfig(**values)


class PowerControllerTuneTests(unittest.TestCase):
    def test_accepts_initial_reading_at_tolerance_boundary(self):
        sleeps: list[float] = []
        transmitter = _FakeTransmitter()
        sensor = _FakePowerSensor([-10.2])

        result = PowerController(sleep_fn=sleeps.append).tune(
            transmitter,
            sensor,
            _config(),
        )

        self.assertAlmostEqual(result.attenuation_db, 30.0)
        self.assertAlmostEqual(result.power_dbm, -10.2)
        self.assertAlmostEqual(result.gap_db, 0.2)
        self.assertEqual(result.trace, (result.trace[0],))
        self.assertEqual(transmitter.set_calls, [(30.0, 3.0)])
        self.assertEqual(sensor.timeouts, [3.0])
        self.assertEqual(sleeps, [0.25])

    def test_accepts_tolerance_boundary_at_a_large_negative_power(self):
        transmitter = _FakeTransmitter()
        sensor = _FakePowerSensor([-100.2])

        result = PowerController(sleep_fn=lambda _: None).tune(
            transmitter,
            sensor,
            _config(target_power_dbm=-100.0),
        )

        self.assertAlmostEqual(result.gap_db, 0.2)
        self.assertEqual(transmitter.set_calls, [(30.0, 3.0)])

    def test_uses_coarse_step_above_one_db_and_fine_step_at_one_db(self):
        transmitter = _FakeTransmitter()
        sensor = _FakePowerSensor([-13.0, -11.0, -10.2])

        result = PowerController(sleep_fn=lambda _: None).tune(
            transmitter,
            sensor,
            _config(),
        )

        self.assertEqual(
            [call[0] for call in transmitter.set_calls],
            [30.0, 29.0, 28.9],
        )
        self.assertEqual(
            [point.gap_db for point in result.trace],
            [3.0, 1.0, result.gap_db],
        )
        self.assertAlmostEqual(result.gap_db, 0.2)

    def test_uses_fine_step_between_tolerance_and_one_db(self):
        transmitter = _FakeTransmitter()
        sensor = _FakePowerSensor([-10.21, -10.1])

        result = PowerController(sleep_fn=lambda _: None).tune(
            transmitter,
            sensor,
            _config(),
        )

        self.assertEqual(
            [call[0] for call in transmitter.set_calls],
            [30.0, 29.9],
        )
        self.assertAlmostEqual(result.gap_db, 0.1)

    def test_restores_last_safe_attenuation_after_target_overshoot(self):
        transmitter = _FakeTransmitter()
        sensor = _FakePowerSensor([-11.0, -9.95])

        with self.assertRaises(PowerControlError) as caught:
            PowerController(sleep_fn=lambda _: None).tune(
                transmitter,
                sensor,
                _config(),
            )

        self.assertEqual(caught.exception.code, "target_overshoot")
        self.assertEqual(caught.exception.last_safe_attenuation_db, 30.0)
        self.assertEqual(caught.exception.last_safe_power_dbm, -11.0)
        self.assertEqual(caught.exception.measured_power_dbm, -9.95)
        self.assertEqual(caught.exception.limit_power_dbm, -10.0)
        self.assertEqual(len(caught.exception.trace), 2)
        self.assertEqual(
            [call[0] for call in transmitter.set_calls],
            [30.0, 29.9, 30.0],
        )
        self.assertTrue(caught.exception.trace[-1].gap_db < 0.0)

    def test_restores_last_safe_attenuation_after_safety_limit(self):
        transmitter = _FakeTransmitter()
        sensor = _FakePowerSensor([-11.0, 0.01])

        with self.assertRaises(PowerControlError) as caught:
            PowerController(sleep_fn=lambda _: None).tune(
                transmitter,
                sensor,
                _config(),
            )

        self.assertEqual(caught.exception.code, "safety_limit_exceeded")
        self.assertEqual(
            [call[0] for call in transmitter.set_calls],
            [30.0, 29.9, 30.0],
        )

    def test_rejects_non_finite_power_and_restores_a_known_safe_value(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                transmitter = _FakeTransmitter()
                sensor = _FakePowerSensor([-11.0, value])

                with self.assertRaises(PowerControlError) as caught:
                    PowerController(sleep_fn=lambda _: None).tune(
                        transmitter,
                        sensor,
                        _config(),
                    )

                self.assertEqual(caught.exception.code, "non_finite_power")
                self.assertTrue(
                    all(
                        math.isfinite(item.power_dbm) for item in caught.exception.trace
                    )
                )
                self.assertEqual(
                    [call[0] for call in transmitter.set_calls],
                    [30.0, 29.9, 30.0],
                )

    def test_rejects_non_real_power_without_treating_it_as_dbm(self):
        for value in (False, np.bool_(True), "-10.0", -10.0 + 0.0j):
            with self.subTest(value=value):
                transmitter = _FakeTransmitter()
                sensor = _FakePowerSensor([-11.0, value])

                with self.assertRaises(PowerControlError) as caught:
                    PowerController(sleep_fn=lambda _: None).tune(
                        transmitter,
                        sensor,
                        _config(),
                    )

                self.assertEqual(caught.exception.code, "invalid_power")
                self.assertEqual(
                    [call[0] for call in transmitter.set_calls],
                    [30.0, 29.9, 30.0],
                )

    def test_rejects_step_below_minimum_without_setting_it(self):
        transmitter = _FakeTransmitter()
        sensor = _FakePowerSensor([-10.3])

        with self.assertRaises(PowerControlError) as caught:
            PowerController(sleep_fn=lambda _: None).tune(
                transmitter,
                sensor,
                _config(initial_attenuation_db=0.05),
            )

        self.assertEqual(caught.exception.code, "attenuation_limit_reached")
        self.assertEqual(transmitter.set_calls, [(0.05, 3.0)])
        self.assertEqual(sensor.timeouts, [3.0])

    def test_allows_an_exact_step_to_minimum(self):
        transmitter = _FakeTransmitter()
        sensor = _FakePowerSensor([-10.3, -10.1])

        result = PowerController(sleep_fn=lambda _: None).tune(
            transmitter,
            sensor,
            _config(initial_attenuation_db=0.1),
        )

        self.assertEqual([call[0] for call in transmitter.set_calls], [0.1, 0.0])
        self.assertEqual(result.attenuation_db, 0.0)

    def test_fails_after_configured_number_of_adjustments(self):
        transmitter = _FakeTransmitter()
        sensor = _FakePowerSensor([-13.0, -12.0])

        with self.assertRaises(PowerControlError) as caught:
            PowerController(sleep_fn=lambda _: None).tune(
                transmitter,
                sensor,
                _config(max_adjustments=1),
            )

        self.assertEqual(caught.exception.code, "max_adjustments_reached")
        self.assertEqual(len(caught.exception.trace), 2)
        self.assertEqual(
            [call[0] for call in transmitter.set_calls],
            [30.0, 29.0],
        )

    def test_cancellation_before_initial_set_has_no_device_side_effect(self):
        transmitter = _FakeTransmitter()
        sensor = _FakePowerSensor([])

        with self.assertRaises(PowerControlCancelled) as caught:
            PowerController(sleep_fn=lambda _: None).tune(
                transmitter,
                sensor,
                _config(),
                cancel_requested=lambda: True,
            )

        self.assertEqual(caught.exception.code, "cancelled")
        self.assertEqual(transmitter.set_calls, [])
        self.assertEqual(sensor.timeouts, [])

    def test_cancellation_after_a_safe_measurement_preserves_trace(self):
        transmitter = _FakeTransmitter()
        sensor = _FakePowerSensor([-11.0])
        states = iter((False, True))

        with self.assertRaises(PowerControlCancelled) as caught:
            PowerController(sleep_fn=lambda _: None).tune(
                transmitter,
                sensor,
                _config(),
                cancel_requested=lambda: next(states),
            )

        self.assertEqual(len(caught.exception.trace), 1)
        self.assertEqual(caught.exception.last_safe_attenuation_db, 30.0)
        self.assertEqual(transmitter.set_calls, [(30.0, 3.0)])


class PowerControllerMonitorTests(unittest.TestCase):
    def test_monitor_returns_target_overshoot_below_absolute_limit(self):
        sensor = _FakePowerSensor([-5.0])

        power_dbm = PowerController(sleep_fn=lambda _: None).monitor(
            sensor,
            _config(),
        )

        self.assertEqual(power_dbm, -5.0)
        self.assertEqual(sensor.timeouts, [3.0])

    def test_monitor_rejects_absolute_limit_and_non_finite_values(self):
        cases = (
            (0.01, "safety_limit_exceeded"),
            (float("nan"), "non_finite_power"),
        )
        for value, expected_code in cases:
            with self.subTest(value=value):
                sensor = _FakePowerSensor([value])
                with self.assertRaises(PowerControlError) as caught:
                    PowerController(sleep_fn=lambda _: None).measure_safe(
                        sensor,
                        _config(),
                    )
                self.assertEqual(caught.exception.code, expected_code)
                if expected_code == "safety_limit_exceeded":
                    self.assertEqual(caught.exception.measured_power_dbm, value)
                    self.assertEqual(caught.exception.limit_power_dbm, 0.0)
                    self.assertIn(str(value), str(caught.exception))

    def test_monitor_rejects_non_real_values_fail_closed(self):
        for value in (False, np.bool_(True), "-10.0", -10.0 + 0.0j):
            with self.subTest(value=value):
                with self.assertRaises(PowerControlError) as caught:
                    PowerController(sleep_fn=lambda _: None).monitor(
                        _FakePowerSensor([value]),
                        _config(),
                    )
                self.assertEqual(caught.exception.code, "invalid_power")

    def test_monitor_accepts_absolute_limit_boundary(self):
        sensor = _FakePowerSensor([0.0])

        result = PowerController(sleep_fn=lambda _: None).monitor(sensor, _config())

        self.assertEqual(result, 0.0)

    def test_monitor_cancels_before_and_after_safe_measurement(self):
        before_sensor = _FakePowerSensor([])
        with self.assertRaises(PowerControlCancelled):
            PowerController(sleep_fn=lambda _: None).monitor(
                before_sensor,
                _config(),
                cancel_requested=lambda: True,
            )
        self.assertEqual(before_sensor.timeouts, [])

        after_sensor = _FakePowerSensor([-10.0])
        states = iter((False, True))
        with self.assertRaises(PowerControlCancelled):
            PowerController(sleep_fn=lambda _: None).monitor(
                after_sensor,
                _config(),
                cancel_requested=lambda: next(states),
            )
        self.assertEqual(after_sensor.timeouts, [3.0])

    def test_rejects_invalid_injected_callables(self):
        with self.assertRaises(TypeError):
            PowerController(sleep_fn=None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            PowerController(sleep_fn=lambda _: None).monitor(
                _FakePowerSensor([-10.0]),
                _config(),
                cancel_requested=True,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
