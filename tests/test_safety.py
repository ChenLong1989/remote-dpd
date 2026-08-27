import json
import unittest

import numpy as np

from remote_dpd.safety import (
    PEAK_LIMIT,
    RMS_GROWTH_FACTOR,
    DigitalSafetyError,
    check_candidate,
    check_reference,
    validate_candidate,
    validate_reference,
)


class DigitalSafetyTests(unittest.TestCase):
    def test_reference_peak_boundary_passes(self):
        reference = np.asarray([1.0 + 0.0j, -1.0j])
        report = validate_reference(reference)
        self.assertTrue(report.passed)
        self.assertEqual(report.reference_peak, PEAK_LIMIT)

    def test_reference_peak_above_limit_fails_without_modification(self):
        reference = np.asarray([1.001 + 0.0j, 0.0j])
        original = reference.copy()
        report = check_reference(reference)
        self.assertFalse(report.passed)
        self.assertIn("reference_peak_exceeded", report.violations)
        np.testing.assert_array_equal(reference, original)
        with self.assertRaises(DigitalSafetyError) as context:
            validate_reference(reference)
        self.assertEqual(context.exception.report.signal_role, "reference")

    def test_candidate_peak_boundary_passes_and_excess_fails(self):
        reference = np.full(4, 0.5 + 0.0j)
        boundary = np.asarray([1.0 + 0.0j, 0.0j, 0.0j, 0.0j])
        self.assertTrue(check_candidate(reference, boundary).passed)

        excessive = boundary.copy()
        excessive[0] = 1.000001
        report = check_candidate(reference, excessive)
        self.assertFalse(report.passed)
        self.assertIn("candidate_peak_exceeded", report.violations)

    def test_candidate_rms_two_db_boundary_passes(self):
        reference = np.full(8, 0.5 + 0.0j)
        candidate = reference * RMS_GROWTH_FACTOR
        report = validate_candidate(reference, candidate)
        self.assertTrue(report.passed)
        self.assertAlmostEqual(report.candidate_rms, report.candidate_rms_limit)

    def test_candidate_rms_above_two_db_fails_without_scaling(self):
        reference = np.full(8, 0.5 + 0.0j)
        candidate = reference * RMS_GROWTH_FACTOR * 1.000001
        original = candidate.copy()
        report = check_candidate(reference, candidate)
        self.assertFalse(report.passed)
        self.assertIn("candidate_rms_exceeded", report.violations)
        np.testing.assert_array_equal(candidate, original)
        with self.assertRaises(DigitalSafetyError):
            validate_candidate(reference, candidate)

    def test_nan_and_inf_are_rejected(self):
        finite = np.ones(2, dtype=np.complex128) * 0.1
        for invalid in (
            np.asarray([np.nan, 0.0], dtype=np.complex128),
            np.asarray([np.inf, 0.0], dtype=np.complex128),
        ):
            with self.subTest(invalid=invalid[0]):
                reference_report = check_reference(invalid)
                self.assertFalse(reference_report.passed)
                self.assertIn("reference_non_finite", reference_report.violations)
                self.assertIsNone(reference_report.reference_peak)
                self.assertIsNone(reference_report.reference_rms)
                json.dumps(reference_report.to_dict(), allow_nan=False)
                candidate_report = check_candidate(finite, invalid)
                self.assertFalse(candidate_report.passed)
                self.assertIn("candidate_non_finite", candidate_report.violations)
                self.assertIsNone(candidate_report.candidate_peak)
                self.assertIsNone(candidate_report.candidate_rms)
                json.dumps(candidate_report.to_dict(), allow_nan=False)

    def test_zero_reference_only_accepts_zero_rms_candidate(self):
        reference = np.zeros(4, dtype=np.complex128)
        zero_report = validate_candidate(reference, reference.copy())
        self.assertEqual(zero_report.candidate_rms_limit, 0.0)

        nonzero = np.asarray([np.finfo(float).tiny, 0.0, 0.0, 0.0])
        report = check_candidate(reference, nonzero)
        self.assertFalse(report.passed)
        self.assertIn("candidate_rms_exceeded", report.violations)

    def test_shape_length_and_numeric_contract_are_reported(self):
        reference = np.ones(4) * 0.1
        cases = (
            (reference.reshape(2, 2), reference, "reference_not_one_dimensional"),
            (reference, reference[:2], "length_mismatch"),
            (reference, np.asarray(["bad"] * 4), "candidate_non_numeric"),
            (reference, np.asarray([], dtype=float), "candidate_empty"),
        )
        for current_reference, candidate, violation in cases:
            with self.subTest(violation=violation):
                report = check_candidate(current_reference, candidate)
                self.assertFalse(report.passed)
                self.assertIn(violation, report.violations)
                json.dumps(report.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
