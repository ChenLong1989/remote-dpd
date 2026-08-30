import asyncio
import json
import os
import tempfile
import time
import unittest
import warnings
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat as scipy_loadmat
from scipy.io import savemat
from starlette.exceptions import StarletteDeprecationWarning
from starlette.requests import Request

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=StarletteDeprecationWarning,
)

from fastapi.testclient import TestClient

import remote_dpd.web as web_module
from remote_dpd.file_interface import FileCommandService
from remote_dpd.storage import RunStore
from remote_dpd.waveforms import WaveformAccessError, WaveformRepository
from remote_dpd.web import create_web_app


def _reference(sample_count: int = 64) -> np.ndarray:
    samples = np.arange(sample_count)
    return (
        0.24 * np.exp(2j * np.pi * 3 * samples / sample_count)
        + 0.08 * np.exp(2j * np.pi * 9 * samples / sample_count)
    ).astype(np.complex128)


def _configuration(sample_count: int = 64) -> dict[str, object]:
    return {
        "device_type": "simulated",
        "device_config": {
            "center_frequency_hz": 3.5e9,
            "sample_rate_hz": 245.76e6,
            "tx_channel": "0",
            "rx_channel": "0",
            "trigger": "immediate",
            "average_segment_count": 2,
            "target_power_dbm": -10.0,
            "safety_power_limit_dbm": 0.0,
            "initial_attenuation_db": 30.0,
            "min_attenuation_db": 0.0,
            "max_attenuation_db": 60.0,
            "settle_seconds": 0.0,
            "max_adjustments": 100,
            "call_timeout_seconds": 1.0,
            "device_options": {
                "noise_dbfs": -100.0,
                "random_seed": 7,
                "power_reference_dbm": 10.0,
                "max_capture_samples": sample_count * 2,
            },
        },
        "runtime_name": "basic_ilc",
        "runtime_config": {"mu": 0.5},
        "max_iterations": 1,
    }


async def _invoke_download_response(
    response: object,
    *,
    fail_on_start: bool = False,
) -> bytes:
    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        raise AssertionError("ASGI receive should not be called for spec 2.4")

    async def send(message: dict[str, object]) -> None:
        if fail_on_start and message["type"] == "http.response.start":
            raise RuntimeError("response start failed")
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
    }
    await response(scope, receive, send)  # type: ignore[operator]
    return b"".join(
        message.get("body", b"")  # type: ignore[arg-type]
        for message in messages
        if message["type"] == "http.response.body"
    )


def _validate_and_replace(
    handle,
    *,
    validator,
    result_path,
    replacement_kind,
    validated_handles,
):
    validated = validator(handle)
    validated_handles.append(handle)
    replacement = result_path.with_name(f"replacement-{replacement_kind}.mat")
    replacement.write_bytes(b"unverified replacement")
    if replacement_kind == "symlink":
        result_path.unlink()
        result_path.symlink_to(replacement.name)
    else:
        os.replace(replacement, result_path)
    return validated


class WebDownloadSecurityRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.waveform_root = self.root / "waveforms"
        self.waveform_root.mkdir()
        self.run_store = RunStore(
            self.root / "runtime",
            retention_seconds=0.0,
        )
        self.command_service = FileCommandService(
            self.root / "exchange",
            run_store=self.run_store,
            status_poll_seconds=0.002,
        )
        self.run_id = "security-regression-run"
        command_path = self.command_service.inbox / f"command_{self.run_id}.mat"
        savemat(
            command_path,
            {
                "schema_version": 1,
                "command_id": self.run_id,
                "action": "run",
                "x": _reference(),
                "config_json": json.dumps(_configuration()),
            },
        )
        status = self.command_service.process_file(command_path)
        self.assertTrue(status.accepted)
        self.assertEqual(status.state, "completed")

        self.app = create_web_app(
            command_service=self.command_service,
            run_store=self.run_store,
            waveform_root=self.waveform_root,
        )
        self.client_context = TestClient(
            self.app,
            base_url="http://127.0.0.1",
        )
        self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.command_service.close()
        self.run_store.close()
        self.temporary.cleanup()

    def _endpoint(self, path: str):
        return next(
            route.endpoint
            for route in self.app.routes
            if getattr(route, "path", None) == path
        )

    def test_run_export_guard_releases_when_response_start_fails(self) -> None:
        endpoint = self._endpoint("/api/v1/runs/{run_id}/result.mat")
        response = asyncio.run(endpoint(self.run_id))
        handle = response._download_handle

        self.assertFalse(handle.closed)
        self.assertEqual(
            self.run_store.cleanup_expired(now=time.time() + 1.0),
            (),
        )
        with self.assertRaisesRegex(RuntimeError, "response start failed"):
            asyncio.run(_invoke_download_response(response, fail_on_start=True))

        self.assertTrue(handle.closed)
        self.assertEqual(
            self.run_store.cleanup_expired(now=time.time() + 1.0),
            (self.run_id,),
        )

    def test_sse_client_slot_releases_when_response_start_fails(self) -> None:
        endpoint = self._endpoint("/api/v1/events")
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/v1/events",
            "raw_path": b"/api/v1/events",
            "query_string": b"",
            "headers": [(b"host", b"127.0.0.1")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 80),
            "root_path": "",
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        response = asyncio.run(endpoint(Request(scope, receive)))
        self.assertEqual(self.app.state.console.sse_clients, 1)

        with self.assertRaisesRegex(RuntimeError, "response start failed"):
            asyncio.run(_invoke_download_response(response, fail_on_start=True))

        self.assertEqual(self.app.state.console.sse_clients, 0)

    def test_command_and_run_downloads_keep_the_validated_descriptor(self) -> None:
        cases = (
            (
                "/api/v1/results/{command_id}.mat",
                self.command_service.result_path(self.run_id),
                "symlink",
            ),
            (
                "/api/v1/runs/{run_id}/result.mat",
                self.run_store.runs_root / self.run_id / "final_result.mat",
                "regular",
            ),
        )
        real_validator = web_module.load_final_payload_file

        for route_path, result_path, replacement_kind in cases:
            with self.subTest(route_path=route_path):
                original = result_path.read_bytes()
                validated_handles = []

                endpoint = self._endpoint(route_path)
                with patch.object(
                    web_module,
                    "load_final_payload_file",
                    side_effect=partial(
                        _validate_and_replace,
                        validator=real_validator,
                        result_path=result_path,
                        replacement_kind=replacement_kind,
                        validated_handles=validated_handles,
                    ),
                ):
                    response = asyncio.run(endpoint(self.run_id))

                handle = response._download_handle
                self.assertIs(validated_handles[0], handle)
                self.assertFalse(handle.closed)
                downloaded = asyncio.run(_invoke_download_response(response))

                self.assertEqual(downloaded, original)
                self.assertTrue(handle.closed)
                self.assertNotEqual(result_path.read_bytes(), original)


class WaveformLoadingSecurityRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_whosmat_rejects_oversized_x_before_loadmat(self) -> None:
        savemat(
            self.root / "oversized.mat",
            {"x": _reference(16), "unrelated": np.ones((16, 16))},
        )
        repository = WaveformRepository(self.root, max_samples=8)
        self.addCleanup(repository.close)

        with (
            patch("scipy.io.loadmat") as loadmat,
            self.assertRaisesRegex(WaveformAccessError, "samples"),
        ):
            repository.load_x("oversized.mat")

        loadmat.assert_not_called()

    def test_loadmat_requests_only_x_when_other_variables_are_present(self) -> None:
        reference = _reference(8)
        savemat(
            self.root / "selective.mat",
            {"x": reference, "unrelated": np.ones((128, 128))},
            do_compression=True,
        )
        repository = WaveformRepository(self.root, max_samples=8)
        self.addCleanup(repository.close)

        with patch("scipy.io.loadmat", wraps=scipy_loadmat) as loadmat:
            loaded = repository.load_x("selective.mat")

        np.testing.assert_allclose(loaded, reference)
        loadmat.assert_called_once()
        self.assertEqual(loadmat.call_args.kwargs["variable_names"], ["x"])


if __name__ == "__main__":
    unittest.main()
