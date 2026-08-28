import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from remote_dpd.file_interface import FileCommandService
from remote_dpd.protocol import save_mat as real_save_mat
from remote_dpd.storage import RunStore
from remote_dpd.waveforms import WaveformRepository
from remote_dpd.web_bridge import WebCommandBridge


class WebStopPriorityTests(unittest.TestCase):
    def test_stop_cancels_a_command_still_preparing_its_waveform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = RunStore(root / "runtime")
            service = FileCommandService(root / "exchange", run_store=store)
            repository = WaveformRepository(root / "waveforms")
            bridge = WebCommandBridge(service, store, repository)
            load_entered = threading.Event()
            release_load = threading.Event()
            submit_errors = []
            stop_results = []

            def gated_load(_path):
                load_entered.set()
                if not release_load.wait(5.0):
                    raise TimeoutError("test waveform load gate timed out")
                return np.asarray([0.2 + 0.0j, -0.2 + 0.0j])

            def submit_command():
                try:
                    bridge.submit(
                        {
                            "action": "load",
                            "waveform_path": "reference.mat",
                            "request_id": "preparing",
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - captured for assertion
                    submit_errors.append(exc)

            with patch.object(repository, "load_x", side_effect=gated_load):
                submit_thread = threading.Thread(target=submit_command)
                submit_thread.start()
                self.assertTrue(load_entered.wait(2.0))

                stop_thread = threading.Thread(
                    target=lambda: stop_results.append(bridge.stop("during-load"))
                )
                stop_thread.start()
                stop_thread.join(2.0)
                self.assertFalse(stop_thread.is_alive())

                release_load.set()
                submit_thread.join(5.0)

            service.close()
            store.close()
            repository.close()
            self.assertFalse(submit_thread.is_alive())
            self.assertTrue(stop_results[0]["accepted"])
            self.assertEqual(len(submit_errors), 1)
            self.assertEqual(submit_errors[0].code, "command_cancelled")
            self.assertFalse(
                (service.inbox / "command_web-command-preparing.mat").exists()
            )

    def test_stop_requests_cancellation_before_waiting_for_command_save(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = RunStore(root / "runtime")
            service = FileCommandService(root / "exchange", run_store=store)
            repository = WaveformRepository(root / "waveforms")
            bridge = WebCommandBridge(service, store, repository)
            save_entered = threading.Event()
            release_save = threading.Event()
            stop_requested = threading.Event()
            submit_result = []
            stop_result = []
            failures = []

            original_request_stop = service.processor.request_stop

            def observed_request_stop():
                stop_requested.set()
                return original_request_stop()

            def gated_save(path, payload):
                if payload["action"] != "stop":
                    save_entered.set()
                    if not release_save.wait(5.0):
                        raise TimeoutError("test command save gate timed out")
                return real_save_mat(path, payload)

            def submit_command():
                try:
                    submit_result.append(
                        bridge.submit({"action": "reset", "request_id": "slow"})
                    )
                except Exception as exc:  # noqa: BLE001 - captured for assertion
                    failures.append(exc)

            def stop_command():
                try:
                    stop_result.append(bridge.stop("urgent"))
                except Exception as exc:  # noqa: BLE001 - captured for assertion
                    failures.append(exc)

            service.processor.request_stop = observed_request_stop
            try:
                with patch("remote_dpd.web_bridge.save_mat", side_effect=gated_save):
                    submit_thread = threading.Thread(target=submit_command)
                    submit_thread.start()
                    self.assertTrue(save_entered.wait(2.0))

                    stop_thread = threading.Thread(target=stop_command)
                    stop_thread.start()
                    self.assertTrue(stop_requested.wait(0.5))
                    self.assertTrue(stop_thread.is_alive())

                    release_save.set()
                    submit_thread.join(5.0)
                    stop_thread.join(5.0)
            finally:
                release_save.set()
                service.processor.request_stop = original_request_stop
                service.close()
                store.close()
                repository.close()

            self.assertFalse(submit_thread.is_alive())
            self.assertFalse(stop_thread.is_alive())
            self.assertEqual(failures, [])
            self.assertFalse(submit_result[0]["accepted"])
            self.assertEqual(submit_result[0]["error"]["code"], "stop_pending")
            self.assertTrue(stop_result[0]["accepted"])

    def test_immediate_stop_barrier_rejects_an_ordinary_file_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = FileCommandService(root / "exchange")
            command_path = service.inbox / "command_pending.mat"
            real_save_mat(
                command_path,
                {
                    "schema_version": 1,
                    "command_id": "pending",
                    "action": "reset",
                },
            )

            with service.immediate_stop_barrier():
                status = service.process_file(command_path)

            self.assertFalse(status.accepted)
            self.assertEqual(status.state, "stopped")
            self.assertEqual(status.error_code, "stop_pending")
            service.close()

    def test_retrying_a_persisted_stop_does_not_request_stop_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = RunStore(root / "runtime")
            service = FileCommandService(root / "exchange", run_store=store)
            repository = WaveformRepository(root / "waveforms")
            bridge = WebCommandBridge(service, store, repository)
            stop_calls = 0
            original_request_stop = service.processor.request_stop

            def counted_request_stop():
                nonlocal stop_calls
                stop_calls += 1
                return original_request_stop()

            service.processor.request_stop = counted_request_stop
            try:
                first = bridge.stop("repeat")
                calls_after_first = stop_calls
                second = bridge.stop("repeat")
            finally:
                service.processor.request_stop = original_request_stop
                service.close()
                store.close()
                repository.close()

            self.assertGreaterEqual(calls_after_first, 1)
            self.assertEqual(stop_calls, calls_after_first)
            self.assertEqual(second["command_id"], first["command_id"])


if __name__ == "__main__":
    unittest.main()
