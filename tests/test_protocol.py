import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat, savemat

from remote_dpd.protocol import save_mat


class MatProtocolTests(unittest.TestCase):
    def test_concurrent_saves_use_distinct_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "result.mat"
            entered = threading.Barrier(2)
            temporary_paths: list[Path] = []
            path_lock = threading.Lock()
            failures: list[BaseException] = []

            def synchronized_savemat(path, values, **options):
                with path_lock:
                    temporary_paths.append(Path(path))
                entered.wait(timeout=5.0)
                savemat(path, values, **options)

            def write(value: int) -> None:
                try:
                    save_mat(target, {"value": np.int64(value)})
                except BaseException as exc:  # noqa: BLE001 - retain thread failure
                    failures.append(exc)

            with patch("scipy.io.savemat", side_effect=synchronized_savemat):
                threads = [
                    threading.Thread(target=write, args=(value,)) for value in (1, 2)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5.0)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(failures, [])
            self.assertEqual(len(temporary_paths), 2)
            self.assertEqual(len(set(temporary_paths)), 2)
            self.assertTrue(all(path.suffix == ".mat" for path in temporary_paths))
            payload = loadmat(target, squeeze_me=True)
            self.assertIn(int(payload["value"]), {1, 2})
            self.assertEqual(list(root.glob(".*.tmp.mat")), [])


if __name__ == "__main__":
    unittest.main()
