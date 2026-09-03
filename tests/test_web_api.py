import asyncio
import json
import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path

import numpy as np
from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=StarletteDeprecationWarning,
)

from fastapi.testclient import TestClient
from scipy.io import loadmat, savemat
from starlette.requests import Request

from remote_dpd.controller import ClosedLoopController
from remote_dpd.file_interface import FileCommandService
from remote_dpd.protocol import save_mat
from remote_dpd.simulation import SimulatedRFBench
from remote_dpd.storage import RunStore
from remote_dpd.web import CONTROL_HEADER, MAX_CONTROL_BODY_BYTES, create_web_app


def _reference(sample_count=64):
    samples = np.arange(sample_count)
    return (
        0.24 * np.exp(2j * np.pi * 3 * samples / sample_count)
        + 0.08 * np.exp(2j * np.pi * 9 * samples / sample_count)
    ).astype(np.complex128)


def _configuration(sample_count=64, max_iterations=2):
    return {
        "device_type": "simulated",
        "normalize_reference_rms": False,
        "reference_target_rms_dbfs": -15.0,
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
        "max_iterations": max_iterations,
    }


def _analysis_profile(**overrides):
    payload = {
        "schema_version": 1,
        "points": 256,
        "frequency_mode": "relative",
        "traces": [
            "baseline_z",
            "target_z",
            "target_error",
            "reference_x",
            "target_y",
        ],
        "bands": [
            {
                "label": "Main",
                "role": "main",
                "center_offset_hz": 20e6,
                "integration_bandwidth_hz": 40e6,
                "enabled": True,
            },
            {
                "label": "Adjacent R1",
                "role": "adjacent",
                "center_offset_hz": 60e6,
                "integration_bandwidth_hz": 20e6,
                "enabled": True,
            },
        ],
    }
    payload.update(overrides)
    return payload


class _GateSimulatedBench(SimulatedRFBench):
    def __init__(self, entered, release):
        super().__init__()
        self._entered = entered
        self._release = release

    def capture(self, request, timeout_seconds):
        self._entered.set()
        if not self._release.wait(5.0):
            raise TimeoutError("test capture gate timed out")
        return super().capture(request, timeout_seconds)


class WebAPITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.waveform_root = self.root / "waveforms"
        self.waveform_root.mkdir()
        savemat(self.waveform_root / "reference.mat", {"x": _reference()})
        self.store = RunStore(self.root / "runtime")
        self.service = FileCommandService(
            self.root / "exchange",
            run_store=self.store,
            status_poll_seconds=0.002,
        )
        self.app = create_web_app(
            command_service=self.service,
            run_store=self.store,
            waveform_root=self.waveform_root,
        )
        self.client_context = TestClient(
            self.app,
            base_url="http://127.0.0.1",
        )
        self.client = self.client_context.__enter__()
        self.control_headers = {
            CONTROL_HEADER: "1",
            "Origin": "http://127.0.0.1",
        }

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.service.close()
        self.store.close()
        self.temporary.cleanup()

    def replace_with_gated_service(self):
        self.client_context.__exit__(None, None, None)
        self.service.close()
        self.store.close()
        entered = threading.Event()
        release = threading.Event()

        def factory(_device_type):
            return ClosedLoopController(_GateSimulatedBench(entered, release))

        self.store = RunStore(self.root / "gated-runtime")
        self.service = FileCommandService(
            self.root / "gated-exchange",
            run_store=self.store,
            controller_factory=factory,
            status_poll_seconds=0.002,
        )
        self.app = create_web_app(
            command_service=self.service,
            run_store=self.store,
            waveform_root=self.waveform_root,
        )
        self.client_context = TestClient(self.app, base_url="http://127.0.0.1")
        self.client = self.client_context.__enter__()
        return entered, release

    def wait_for_web_command(self, command_id):
        deadline = time.monotonic() + 5.0
        while True:
            response = self.client.get(f"/api/v1/commands/{command_id}")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            if payload["phase"] in {"completed", "failed", "stopped", "rejected"}:
                return payload
            if time.monotonic() >= deadline:
                self.fail("Web command did not reach a terminal phase")
            time.sleep(0.01)

    def test_shell_health_devices_waveforms_and_security_headers(self):
        page = self.client.get("/")
        styles = self.client.get("/static/styles.css?v=single-screen-7")
        script = self.client.get("/static/app.js?v=single-screen-7")
        health = self.client.get("/api/v1/health")
        devices = self.client.get("/api/v1/devices")
        waveforms = self.client.get("/api/v1/waveforms")
        preview = self.client.get(
            "/api/v1/waveforms/preview",
            params={"path": "reference.mat", "points": 32},
        )

        self.assertEqual(page.status_code, 200)
        self.assertIn("Remote DPD RF Workbench", page.text)
        self.assertIn('class="single-screen"', page.text)
        self.assertIn('id="configuration-dialog"', page.text)
        self.assertIn('id="expert-dialog"', page.text)
        self.assertIn('id="runs-dialog"', page.text)
        self.assertIn('id="run-confirm-dialog"', page.text)
        self.assertIn("single-screen-7", page.text)
        self.assertIn('value="target_error" checked', page.text)
        self.assertIn('data-aux-view="aclr"', page.text)
        self.assertIn('id="normalize-reference-rms"', page.text)
        self.assertIn('id="reference-target-rms"', page.text)
        self.assertIn('id="runtime-name"', page.text)
        self.assertIn('value="forward_model_ilc"', page.text)
        self.assertIn('byId("runtime-name").value', script.text)
        self.assertEqual(health.json()["status"], "ok")
        device_types = [entry["device_type"] for entry in devices.json()["devices"]]
        self.assertIn("simulated", device_types)
        self.assertIn("vst5842", device_types)
        simulated = next(
            entry
            for entry in devices.json()["devices"]
            if entry["device_type"] == "simulated"
        )
        self.assertEqual(simulated["device_type"], "simulated")
        self.assertEqual(simulated["schema"]["schema_version"], 3)
        default_configuration = simulated["default_configuration"]
        default_common = default_configuration["device_config"]
        self.assertTrue(default_configuration["normalize_reference_rms"])
        self.assertEqual(default_configuration["reference_target_rms_dbfs"], -15.0)
        self.assertEqual(default_common["target_power_dbm"], -15.0)
        self.assertEqual(default_common["sample_rate_hz"], 491.52e6)
        self.assertEqual(default_common["average_segment_count"], 10)
        self.assertEqual(
            default_common["device_options"]["max_capture_samples"],
            10_000_000,
        )
        self.assertEqual(
            default_common["device_options"]["power_reference_dbm"],
            1.0,
        )
        self.assertEqual(
            [
                row
                for row in default_common["device_options"]["pa_coefficients"]
                if row["p"] == 3
            ],
            [
                {"p": 3, "m": 0, "real": -1.44, "imag": 0.3},
                {"p": 3, "m": 1, "real": -0.24, "imag": 0.12},
            ],
        )
        noise_field = next(
            field
            for field in simulated["schema"]["fields"]
            if field["name"] == "noise_dbfs"
        )
        self.assertEqual(noise_field["default"], -85.74)
        self.assertEqual(default_configuration["max_iterations"], 15)
        self.assertEqual(default_configuration["runtime_name"], "forward_model_ilc")
        self.assertEqual(default_configuration["runtime_config"]["mu"], 1.0)
        schema_capture = next(
            field
            for field in simulated["schema"]["fields"]
            if field["name"] == "max_capture_samples"
        )
        self.assertEqual(schema_capture["default"], 1_000_000)
        self.assertEqual(waveforms.json()["entries"][0]["path"], "reference.mat")
        self.assertEqual(preview.json()["preview_count"], 32)
        self.assertEqual(health.headers["cache-control"], "no-store")
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertEqual(styles.headers["cache-control"], "no-store")
        self.assertEqual(script.headers["cache-control"], "no-store")
        self.assertIn("crypto?.getRandomValues", script.text)
        self.assertNotIn("randomUUID", script.text)
        self.assertEqual(page.headers["x-frame-options"], "DENY")
        self.assertNotIn("access-control-allow-origin", health.headers)

    def test_explicit_lan_host_is_allowed_and_other_hosts_remain_rejected(self):
        lan_app = create_web_app(
            command_service=self.service,
            run_store=self.store,
            waveform_root=self.waveform_root,
            allowed_hosts=["192.168.3.100"],
        )
        with TestClient(
            lan_app,
            base_url="http://192.168.3.100:8765",
        ) as lan_client:
            health = lan_client.get("/api/v1/health")
            accepted = lan_client.post(
                "/api/v1/commands",
                json={"action": "reset", "request_id": "lan-reset"},
                headers={
                    CONTROL_HEADER: "1",
                    "Origin": "http://192.168.3.100:8765",
                },
            )
            wrong_host = lan_client.get(
                "/api/v1/health",
                headers={"Host": "192.168.3.101:8765"},
            )
            wrong_origin = lan_client.post(
                "/api/v1/commands",
                json={"action": "reset"},
                headers={
                    CONTROL_HEADER: "1",
                    "Origin": "http://192.168.3.101:8765",
                },
            )

        self.assertEqual(health.status_code, 200)
        self.assertEqual(accepted.status_code, 202)
        self.assertEqual(wrong_host.status_code, 400)
        self.assertEqual(wrong_origin.status_code, 403)

    def test_web_app_rejects_unsafe_additional_hosts(self):
        for host in ("*", "0.0.0.0", "8.8.8.8", "bad host"):
            with self.subTest(host=host), self.assertRaises((TypeError, ValueError)):
                create_web_app(
                    command_service=self.service,
                    run_store=self.store,
                    waveform_root=self.waveform_root,
                    allowed_hosts=[host],
                )

    def test_control_requests_enforce_host_origin_header_json_and_size(self):
        payload = {"action": "reset"}

        missing_header = self.client.post("/api/v1/commands", json=payload)
        wrong_type = self.client.post(
            "/api/v1/commands",
            content=json.dumps(payload),
            headers={CONTROL_HEADER: "1", "Content-Type": "text/plain"},
        )
        wrong_origin = self.client.post(
            "/api/v1/commands",
            json=payload,
            headers={CONTROL_HEADER: "1", "Origin": "https://evil.example"},
        )
        null_origin = self.client.post(
            "/api/v1/commands",
            json=payload,
            headers={CONTROL_HEADER: "1", "Origin": "null"},
        )
        wrong_host = self.client.get(
            "/api/v1/health",
            headers={"Host": "evil.example"},
        )
        oversized = self.client.post(
            "/api/v1/commands",
            content=b"{" + b" " * MAX_CONTROL_BODY_BYTES + b"}",
            headers={CONTROL_HEADER: "1", "Content-Type": "application/json"},
        )

        def oversized_chunks():
            for _ in range(17):
                yield b"x" * 65536

        chunked = self.client.post(
            "/api/v1/commands",
            content=oversized_chunks(),
            headers={CONTROL_HEADER: "1", "Content-Type": "application/json"},
        )

        self.assertEqual(missing_header.status_code, 403)
        self.assertEqual(wrong_type.status_code, 415)
        self.assertEqual(wrong_origin.status_code, 403)
        self.assertEqual(null_origin.status_code, 403)
        self.assertEqual(wrong_host.status_code, 400)
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(chunked.status_code, 413)

    def test_control_json_rejects_duplicates_constants_and_unknown_fields(self):
        duplicate = self.client.post(
            "/api/v1/commands",
            content=b'{"action":"reset","action":"run"}',
            headers={**self.control_headers, "Content-Type": "application/json"},
        )
        constant = self.client.post(
            "/api/v1/commands",
            content=b'{"action":"run","configuration":{"value":NaN}}',
            headers={**self.control_headers, "Content-Type": "application/json"},
        )
        unknown = self.client.post(
            "/api/v1/commands",
            json={"action": "reset", "server_path": "/tmp/result.mat"},
            headers=self.control_headers,
        )

        self.assertEqual(duplicate.status_code, 400)
        self.assertEqual(constant.status_code, 400)
        self.assertEqual(unknown.status_code, 422)

    def test_web_configuration_operation_limits_are_enforced(self):
        configuration = _configuration()
        configuration["max_iterations"] = 1001
        iterations = self.client.post(
            "/api/v1/commands",
            json={"action": "configure", "configuration": configuration},
            headers=self.control_headers,
        )
        configuration = _configuration()
        configuration["device_config"]["device_options"]["pa_coefficients"] = [
            {"p": 1, "m": 0, "real": 1.0, "imag": 0.0}
        ] * 257
        coefficients = self.client.post(
            "/api/v1/commands",
            json={"action": "configure", "configuration": configuration},
            headers=self.control_headers,
        )

        self.assertEqual(iterations.status_code, 422)
        self.assertEqual(
            iterations.json()["error"]["code"],
            "configuration_limit_exceeded",
        )
        self.assertEqual(coefficients.status_code, 422)

    def test_analysis_requests_use_control_security_and_are_strictly_bounded(self):
        missing_header = self.client.post(
            "/api/v1/session/analysis",
            json=_analysis_profile(),
        )
        unknown = self.client.post(
            "/api/v1/session/analysis",
            json=_analysis_profile(server_path="/tmp/data"),
            headers=self.control_headers,
        )
        too_many_bands = self.client.post(
            "/api/v1/session/analysis",
            json=_analysis_profile(
                bands=[
                    {
                        "label": f"Band {index}",
                        "role": "other",
                        "center_offset_hz": 0.0,
                        "integration_bandwidth_hz": 1.0,
                    }
                    for index in range(33)
                ]
            ),
            headers=self.control_headers,
        )
        missing_configuration = self.client.post(
            "/api/v1/session/analysis",
            json=_analysis_profile(),
            headers=self.control_headers,
        )

        self.assertEqual(missing_header.status_code, 403)
        self.assertEqual(unknown.status_code, 422)
        self.assertEqual(too_many_bands.status_code, 413)
        self.assertEqual(missing_configuration.status_code, 409)
        self.assertEqual(
            missing_configuration.json()["error"]["code"], "reference_missing"
        )

    def test_request_id_retries_are_exact_and_stop_uses_a_separate_namespace(self):
        first = self.client.post(
            "/api/v1/commands",
            json={"action": "reset", "request_id": "shared-key"},
            headers=self.control_headers,
        )
        first_status = self.wait_for_web_command(first.json()["command_id"])
        retry = self.client.post(
            "/api/v1/commands",
            json={"action": "reset", "request_id": "shared-key"},
            headers=self.control_headers,
        )
        conflict = self.client.post(
            "/api/v1/commands",
            json={
                "action": "load",
                "waveform_path": "reference.mat",
                "request_id": "shared-key",
            },
            headers=self.control_headers,
        )
        stopped = self.client.post(
            "/api/v1/stop",
            json={"request_id": "shared-key"},
            headers=self.control_headers,
        )

        self.assertEqual(first_status["phase"], "completed")
        self.assertEqual(retry.status_code, 202)
        self.assertEqual(retry.json()["command_id"], first.json()["command_id"])
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(
            conflict.json()["error"]["code"],
            "idempotency_conflict",
        )
        self.assertEqual(stopped.status_code, 200)
        self.assertNotEqual(stopped.json()["command_id"], first.json()["command_id"])
        self.assertTrue(stopped.json()["command_id"].startswith("web-stop-"))

    def test_invalid_run_and_command_identifiers_do_not_escape_storage(self):
        command = self.client.get("/api/v1/commands/..")
        run = self.client.get("/api/v1/runs/..")
        result = self.client.get("/api/v1/results/...mat")

        self.assertEqual(command.status_code, 404)
        self.assertEqual(run.status_code, 404)
        self.assertEqual(result.status_code, 404)

    def test_web_run_completes_through_shared_coordinator_and_exposes_artifacts(self):
        submitted = self.client.post(
            "/api/v1/commands",
            json={
                "action": "run",
                "waveform_path": "reference.mat",
                "configuration": _configuration(),
                "request_id": "e2e-run",
            },
            headers=self.control_headers,
        )
        self.assertEqual(submitted.status_code, 202)
        command_id = submitted.json()["command_id"]
        payload = self.wait_for_web_command(command_id)
        self.assertEqual(payload["phase"], "completed")
        self.assertEqual(payload["run_id"], command_id)
        self.assertIsNotNone(payload["result_url"])

        session = self.client.get("/api/v1/session").json()
        current_preview = self.client.get(
            "/api/v1/session/preview",
            params={"points": 32},
        ).json()
        runs = self.client.get("/api/v1/runs").json()
        detail = self.client.get(f"/api/v1/runs/{command_id}").json()
        iteration = self.client.get(
            f"/api/v1/runs/{command_id}/iterations/2/preview",
            params={"points": 32},
        ).json()
        current_analysis_response = self.client.post(
            "/api/v1/session/analysis",
            json=_analysis_profile(),
            headers=self.control_headers,
        )
        run_analysis_response = self.client.post(
            f"/api/v1/runs/{command_id}/analysis",
            json=_analysis_profile(baseline_iteration=0, target_iteration=2),
            headers=self.control_headers,
        )
        session_after_analysis = self.client.get("/api/v1/session").json()
        result = self.client.get(payload["result_url"])
        run_result = self.client.get(f"/api/v1/runs/{command_id}/result.mat")

        self.assertEqual(session["controller"]["state"], "completed")
        self.assertEqual(len(session["controller"]["records"]), 3)
        self.assertFalse(session["controller"]["reference_normalization"]["enabled"])
        self.assertEqual(
            session["controller"]["reference_normalization"]["scale_db"],
            0.0,
        )
        self.assertEqual(current_preview["preview_count"], 32)
        self.assertEqual(runs["runs"][0]["run_id"], command_id)
        self.assertEqual(detail["run"]["status"], "completed")
        self.assertEqual(iteration["preview_count"], 32)
        self.assertEqual(current_analysis_response.status_code, 200)
        self.assertEqual(run_analysis_response.status_code, 200)
        current_analysis = current_analysis_response.json()
        run_analysis = run_analysis_response.json()
        self.assertEqual(current_analysis["fft_size"], 64)
        self.assertEqual(current_analysis["comparison"]["baseline"]["iteration"], 0)
        self.assertEqual(current_analysis["comparison"]["target"]["iteration"], 2)
        self.assertIn(
            "target_error",
            [trace["key"] for trace in current_analysis["traces"]],
        )
        self.assertEqual(run_analysis["comparison"]["target"]["iteration"], 2)
        self.assertLessEqual(len(run_analysis["frequency_hz"]), 256)
        self.assertEqual(run_analysis_response.headers["cache-control"], "no-store")
        self.assertEqual(session_after_analysis, session)
        self.assertNotIn("x", detail["run"])
        self.assertEqual(result.status_code, 200)
        self.assertEqual(run_result.status_code, 200)

        saved_result = self.root / "downloaded.mat"
        saved_result.write_bytes(result.content)
        loaded = loadmat(saved_result, squeeze_me=True)
        self.assertEqual(str(loaded["status"]), "completed")

    def test_stepwise_web_actions_use_action_specific_terminal_phases(self):
        loaded = self.client.post(
            "/api/v1/commands",
            json={"action": "load", "waveform_path": "reference.mat"},
            headers=self.control_headers,
        )
        loaded_status = self.wait_for_web_command(loaded.json()["command_id"])
        configured = self.client.post(
            "/api/v1/commands",
            json={"action": "configure", "configuration": _configuration()},
            headers=self.control_headers,
        )
        configured_status = self.wait_for_web_command(configured.json()["command_id"])
        started = self.client.post(
            "/api/v1/commands",
            json={"action": "start_transmission"},
            headers=self.control_headers,
        )
        started_status = self.wait_for_web_command(started.json()["command_id"])
        stopped_tx = self.client.post(
            "/api/v1/commands",
            json={"action": "stop_transmission"},
            headers=self.control_headers,
        )
        stopped_tx_status = self.wait_for_web_command(stopped_tx.json()["command_id"])
        power = self.client.post(
            "/api/v1/commands",
            json={"action": "power_tune"},
            headers=self.control_headers,
        )
        power_status = self.wait_for_web_command(power.json()["command_id"])

        self.assertEqual(loaded_status["phase"], "completed")
        self.assertIn(loaded_status["controller_state"], {"loaded", "ready"})
        self.assertEqual(configured_status["phase"], "completed")
        self.assertEqual(configured_status["controller_state"], "ready")
        self.assertEqual(started_status["phase"], "completed")
        self.assertEqual(started_status["controller_state"], "ready")
        self.assertEqual(stopped_tx_status["phase"], "completed")
        self.assertEqual(stopped_tx_status["controller_state"], "ready")
        self.assertEqual(power_status["phase"], "completed")
        self.assertEqual(power_status["controller_state"], "power_ready")

    def test_invalid_waveform_does_not_change_session(self):
        savemat(self.waveform_root / "unsafe.mat", {"x": np.zeros(16)})
        before = self.client.get("/api/v1/session").json()

        rejected = self.client.post(
            "/api/v1/commands",
            json={"action": "load", "waveform_path": "unsafe.mat"},
            headers=self.control_headers,
        )
        after = self.client.get("/api/v1/session").json()

        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(before, after)
        self.assertEqual(self.store.list_runs(), ())

    def test_sse_sends_bounded_state_and_releases_client_on_disconnect(self):
        route = next(
            route
            for route in self.app.routes
            if getattr(route, "path", None) == "/api/v1/events"
        )
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
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

        async def read_one_event():
            response = await route.endpoint(Request(scope, receive))
            event = await anext(response.body_iterator)
            await response.body_iterator.aclose()
            return event

        event = asyncio.run(read_one_event())
        text = event.decode() if isinstance(event, bytes) else event
        data_line = next(
            line for line in text.splitlines() if line.startswith("data: ")
        )
        payload = json.loads(data_line.removeprefix("data: "))

        self.assertIn("event: state", text)
        self.assertEqual(payload["schema_version"], 1)
        self.assertNotIn("x", payload)
        self.assertNotIn("y", payload)
        self.assertNotIn("z", payload)
        self.assertEqual(self.app.state.console.sse_clients, 0)

    def test_file_run_blocks_web_command_and_web_stop_cancels_it(self):
        entered, release = self.replace_with_gated_service()
        command_path = self.service.inbox / "command_file-run.mat"
        save_mat(
            command_path,
            {
                "schema_version": 1,
                "command_id": "file-run",
                "action": "run",
                "x": _reference(),
                "config_json": json.dumps(_configuration(max_iterations=5)),
            },
        )
        accepted = self.service.process_file(command_path, background=True)
        self.assertTrue(accepted.accepted)
        self.assertTrue(entered.wait(5.0))

        busy = self.client.post(
            "/api/v1/commands",
            json={"action": "reset"},
            headers=self.control_headers,
        )
        stopped = self.client.post(
            "/api/v1/stop",
            json={},
            headers=self.control_headers,
        )
        release.set()

        deadline = time.monotonic() + 5.0
        while self.service.read_status("file-run").state not in {
            "completed",
            "failed",
            "stopped",
        }:
            if time.monotonic() >= deadline:
                self.fail("file run did not stop")
            time.sleep(0.01)

        self.assertEqual(busy.status_code, 409)
        self.assertEqual(busy.json()["error"]["code"], "busy")
        self.assertEqual(stopped.status_code, 200)
        self.assertEqual(self.service.read_status("file-run").state, "stopped")
        self.assertFalse(self.service.result_path("file-run").exists())

    def test_web_run_blocks_file_command_and_file_stop_cancels_it(self):
        entered, release = self.replace_with_gated_service()
        submitted = self.client.post(
            "/api/v1/commands",
            json={
                "action": "run",
                "waveform_path": "reference.mat",
                "configuration": _configuration(max_iterations=5),
                "request_id": "web-gated-run",
            },
            headers=self.control_headers,
        )
        command_id = submitted.json()["command_id"]
        self.assertTrue(entered.wait(5.0))

        load_path = self.service.inbox / "command_file-load.mat"
        save_mat(
            load_path,
            {
                "schema_version": 1,
                "command_id": "file-load",
                "action": "load",
                "x": _reference(),
            },
        )
        busy = self.service.process_file(load_path)
        stop_path = self.service.inbox / "command_file-stop.mat"
        save_mat(
            stop_path,
            {
                "schema_version": 1,
                "command_id": "file-stop",
                "action": "stop",
            },
        )
        stopped = self.service.process_file(stop_path)
        release.set()

        deadline = time.monotonic() + 5.0
        while True:
            status = self.client.get(f"/api/v1/commands/{command_id}").json()
            if status["phase"] in {"completed", "failed", "stopped"}:
                break
            if time.monotonic() >= deadline:
                self.fail("Web run did not stop")
            time.sleep(0.01)

        self.assertFalse(busy.accepted)
        self.assertEqual(busy.error_code, "busy")
        self.assertIn(stopped.state, {"stopping", "stopped"})
        self.assertEqual(status["phase"], "stopped")


if __name__ == "__main__":
    unittest.main()
