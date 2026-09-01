import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.sparse import csr_matrix

from remote_dpd.waveforms import (
    WaveformAccessError,
    WaveformRepository,
    _validate_hdf5_dataset_layout,
)


def _reference(sample_count=128):
    samples = np.arange(sample_count)
    return (
        0.28 * np.exp(2j * np.pi * 3 * samples / sample_count)
        + 0.07 * np.exp(2j * np.pi * 11 * samples / sample_count)
    ).astype(np.complex128)


class WaveformRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "waveforms"
        self.root.mkdir()
        self.repository = WaveformRepository(self.root)

    def tearDown(self):
        self.repository.close()
        self.temporary.cleanup()

    def test_lists_only_real_directories_and_mat_files(self):
        nested = self.root / "nested"
        nested.mkdir()
        savemat(self.root / "x.mat", {"x": _reference()})
        (self.root / "notes.txt").write_text("ignored", encoding="utf-8")
        (self.root / ".hidden.mat").write_bytes(b"ignored")
        (self.root / "linked.mat").symlink_to(self.root / "x.mat")

        entries = self.repository.list_directory()

        self.assertEqual(
            [(entry.name, entry.kind) for entry in entries],
            [("nested", "directory"), ("x.mat", "waveform")],
        )
        self.assertEqual(self.repository.list_directory("nested"), ())

    def test_loads_row_column_and_complex_vectors_as_read_only_complex128(self):
        reference = _reference()
        savemat(self.root / "row.mat", {"x": reference.reshape(1, -1)})
        savemat(self.root / "column.mat", {"x": reference.reshape(-1, 1)})

        row = self.repository.load_x("row.mat")
        column = self.repository.load_x("column.mat")

        np.testing.assert_allclose(row, reference)
        np.testing.assert_allclose(column, reference)
        self.assertEqual(row.dtype, np.complex128)
        self.assertFalse(row.flags.writeable)
        self.assertFalse(column.flags.writeable)

    def test_preview_is_bounded_and_contains_safety_metadata(self):
        savemat(self.root / "preview.mat", {"x": _reference(1000)})

        preview = self.repository.preview("preview.mat", points=64)

        self.assertEqual(preview["sample_count"], 1000)
        self.assertEqual(preview["preview_count"], 64)
        self.assertEqual(len(preview["magnitude"]), 64)
        self.assertTrue(preview["safety"]["passed"])

    def test_rejects_traversal_absolute_non_normalized_and_symlink_paths(self):
        outside = Path(self.temporary.name) / "outside.mat"
        savemat(outside, {"x": _reference()})
        (self.root / "file-link.mat").symlink_to(outside)
        (self.root / "dir-link").symlink_to(outside.parent, target_is_directory=True)

        unsafe = (
            "../outside.mat",
            str(outside),
            "dir-link/outside.mat",
            "file-link.mat",
            "nested/../outside.mat",
            "nested//x.mat",
            "nested\\x.mat",
            "C:/outside.mat",
            "bad\x00name.mat",
        )
        for path in unsafe:
            with self.subTest(path=path), self.assertRaises(WaveformAccessError):
                self.repository.load_x(path)

    def test_root_descriptor_does_not_follow_a_replaced_root_path(self):
        savemat(self.root / "original.mat", {"x": _reference()})
        original_root = self.root.with_name("original-root")
        self.root.rename(original_root)
        self.root.mkdir()
        savemat(self.root / "replacement.mat", {"x": _reference() * 0.5})

        entries = self.repository.list_directory()

        self.assertEqual([entry.name for entry in entries], ["original.mat"])

    def test_invalid_mat_and_waveform_types_are_rejected(self):
        invalid_values = {
            "missing.mat": {"other": _reference()},
            "scalar.mat": {"x": np.asarray(0.5)},
            "matrix.mat": {"x": np.ones((2, 2))},
            "logical.mat": {"x": np.asarray([True, False])},
            "nan.mat": {"x": np.asarray([0.1, np.nan])},
            "zero.mat": {"x": np.zeros(16)},
            "sparse.mat": {"x": csr_matrix(np.eye(4))},
        }
        for name, payload in invalid_values.items():
            savemat(self.root / name, payload)

        for name in invalid_values:
            with self.subTest(name=name), self.assertRaises(WaveformAccessError):
                self.repository.load_x(name)

        (self.root / "broken.mat").write_bytes(b"not a MAT file")
        with self.assertRaises(WaveformAccessError):
            self.repository.load_x("broken.mat")

    def test_source_peak_above_full_scale_is_reported_but_not_rejected(self):
        savemat(self.root / "source-peak.mat", {"x": np.asarray([1.5, 0.1])})

        loaded = self.repository.load_x("source-peak.mat")
        preview = self.repository.preview("source-peak.mat", points=16)

        np.testing.assert_array_equal(loaded, np.asarray([1.5, 0.1]))
        self.assertFalse(preview["safety"]["passed"])
        self.assertIn(
            "reference_peak_exceeded",
            preview["safety"]["violations"],
        )

    def test_file_and_sample_limits_are_enforced(self):
        small_file_repository = WaveformRepository(self.root, max_bytes=32)
        sample_limited_repository = WaveformRepository(self.root, max_samples=8)
        self.addCleanup(small_file_repository.close)
        self.addCleanup(sample_limited_repository.close)
        savemat(self.root / "limited.mat", {"x": _reference(16)})

        with self.assertRaisesRegex(WaveformAccessError, "bytes"):
            small_file_repository.load_x("limited.mat")
        with self.assertRaisesRegex(WaveformAccessError, "samples"):
            sample_limited_repository.load_x("limited.mat")

    def test_hdf5_layout_rejects_oversized_chunks_and_extensible_shapes(self):
        _validate_hdf5_dataset_layout(
            shape=(1, 8),
            dtype=np.dtype(np.complex128),
            maxshape=(1, 8),
            chunks=(1, 8),
            max_samples=8,
        )
        with self.assertRaisesRegex(WaveformAccessError, "storage chunk"):
            _validate_hdf5_dataset_layout(
                shape=(1, 8),
                dtype=np.dtype(np.complex128),
                maxshape=(1, 8),
                chunks=(1, 1_024),
                max_samples=8,
            )
        with self.assertRaisesRegex(WaveformAccessError, "fixed dataset shape"):
            _validate_hdf5_dataset_layout(
                shape=(1, 8),
                dtype=np.dtype(np.float64),
                maxshape=(None, 8),
                chunks=(1, 8),
                max_samples=8,
            )
        if np.dtype(np.longdouble).itemsize > 8:
            with self.assertRaisesRegex(WaveformAccessError, "numeric dtype"):
                _validate_hdf5_dataset_layout(
                    shape=(1, 8),
                    dtype=np.dtype(np.longdouble),
                    maxshape=(1, 8),
                    chunks=None,
                    max_samples=8,
                )

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO requires POSIX")
    def test_fifo_with_mat_suffix_is_never_listed_or_opened(self):
        fifo = self.root / "capture.mat"
        os.mkfifo(fifo)

        self.assertEqual(self.repository.list_directory(), ())
        with self.assertRaises(WaveformAccessError):
            self.repository.load_x("capture.mat")


if __name__ == "__main__":
    unittest.main()
