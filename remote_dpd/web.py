"""Local-only FastAPI console for the shared closed-loop command service."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .file_interface import FileCommandError
from .result_export import ResultExportError, load_final_payload_file
from .storage import RunNotFoundError
from .waveforms import WaveformAccessError, WaveformRepository
from .web_bridge import WebBridgeError, WebCommandBridge

if TYPE_CHECKING:
    from .file_interface import FileCommandService
    from .storage import RunStore


WEB_API_VERSION = 1
CONTROL_HEADER = "X-Remote-DPD-Request"
MAX_CONTROL_BODY_BYTES = 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 100_000
MAX_SSE_CLIENTS = 8
SSE_STATE_REFRESH_SECONDS = 0.5
_STATIC_ROOT = Path(str(files("remote_dpd").joinpath("web_static")))


class WebAPIError(ValueError):
    """Stable API error with an explicit HTTP status."""

    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class _AnchoredDirectory:
    """Open regular files relative to one pinned real directory."""

    def __init__(self, path: str | Path) -> None:
        raw_path = Path(path)
        if raw_path.is_symlink() or not raw_path.is_dir():
            raise ValueError("result directory must be a real directory")
        flags = os.O_RDONLY | _required_flag("O_DIRECTORY")
        flags |= _required_flag("O_NOFOLLOW") | _optional_flag("O_CLOEXEC")
        self._fd = os.open(raw_path, flags)
        if not stat.S_ISDIR(os.fstat(self._fd).st_mode):  # pragma: no cover
            os.close(self._fd)
            raise ValueError("result directory must be a real directory")
        self._closed = False

    def open_file(self, *components: str) -> BinaryIO:
        if not components or any(
            not _safe_path_component(component) for component in components
        ):
            raise ValueError("result path must contain safe path components")
        if self._closed:
            raise RuntimeError("result directory is closed")
        directory_fd = os.dup(self._fd)
        descriptor: int | None = None
        try:
            directory_flags = os.O_RDONLY | _required_flag("O_DIRECTORY")
            directory_flags |= _required_flag("O_NOFOLLOW")
            directory_flags |= _optional_flag("O_CLOEXEC")
            for component in components[:-1]:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            file_flags = os.O_RDONLY | _required_flag("O_NOFOLLOW")
            file_flags |= _optional_flag("O_CLOEXEC")
            file_flags |= _optional_flag("O_NONBLOCK")
            descriptor = os.open(components[-1], file_flags, dir_fd=directory_fd)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("result must be a regular file")
            handle = os.fdopen(descriptor, "rb", closefd=True)
            descriptor = None
            return handle
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            raise
        finally:
            os.close(directory_fd)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._fd)

    def __enter__(self) -> _AnchoredDirectory:  # noqa: PYI034 - Python 3.10
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class _GuardedDownloadResponse(StreamingResponse):
    """Release an opened result and its guard on every ASGI exit path."""

    def __init__(
        self,
        handle: BinaryIO,
        *,
        filename: str,
        cleanup: Callable[[], None],
    ) -> None:
        self._download_handle = handle
        self._cleanup = cleanup
        self._released = False
        size = os.fstat(handle.fileno()).st_size
        super().__init__(
            self._stream_file(),
            media_type="application/x-matlab-data",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(size),
                "Cache-Control": "no-store",
            },
        )

    async def _stream_file(self) -> AsyncIterator[bytes]:
        while chunk := await run_in_threadpool(
            self._download_handle.read,
            1024 * 1024,
        ):
            yield chunk

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()

    def _release(self) -> None:
        if self._released:
            return
        self._released = True
        self._cleanup()


class _GuardedEventStreamResponse(StreamingResponse):
    """Release an SSE client slot even before its generator starts."""

    def __init__(self, *args: Any, cleanup: Callable[[], None], **kwargs: Any) -> None:
        self._cleanup = cleanup
        self._released = False
        super().__init__(*args, **kwargs)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.release()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._cleanup()


@dataclass(slots=True)
class WebConsoleContext:
    """Resources shared by every route in one local console instance."""

    command_service: FileCommandService
    run_store: RunStore
    waveforms: WaveformRepository
    bridge: WebCommandBridge
    outbox_directory: _AnchoredDirectory
    runs_directory: _AnchoredDirectory
    sse_clients: int = 0
    _sse_cache_lock: threading.Lock = field(default_factory=threading.Lock)
    _sse_cache_deadline: float = 0.0
    _sse_cache_payload: str = ""

    def event_state_json(self) -> str:
        """Build at most one shared state payload per refresh interval."""
        now = time.monotonic()
        with self._sse_cache_lock:
            if now < self._sse_cache_deadline:
                return self._sse_cache_payload
            payload = json.dumps(
                self.bridge.session(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            self._sse_cache_payload = payload
            self._sse_cache_deadline = now + SSE_STATE_REFRESH_SECONDS
            return payload


def create_web_app(
    *,
    command_service: FileCommandService,
    run_store: RunStore,
    waveform_root: str | Path,
) -> FastAPI:
    """Create a loopback console around the same coordinator as MAT commands."""
    resources = ExitStack()
    try:
        repository = WaveformRepository(waveform_root)
        resources.callback(repository.close)
        outbox_directory = resources.enter_context(
            _AnchoredDirectory(command_service.outbox)
        )
        runs_directory = resources.enter_context(
            _AnchoredDirectory(run_store.runs_root)
        )
    except Exception:
        resources.close()
        raise
    bridge = WebCommandBridge(command_service, run_store, repository)
    context = WebConsoleContext(
        command_service,
        run_store,
        repository,
        bridge,
        outbox_directory,
        runs_directory,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            resources.close()

    app = FastAPI(
        title="Remote DPD Control Deck",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.console = context
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(WebAPIError)
    async def web_api_error(_request: Request, exc: WebAPIError) -> JSONResponse:
        return _error_response(exc.code, str(exc), exc.status_code)

    @app.exception_handler(WebBridgeError)
    async def bridge_error(_request: Request, exc: WebBridgeError) -> JSONResponse:
        return _error_response(exc.code, str(exc), exc.status_code)

    @app.exception_handler(WaveformAccessError)
    async def waveform_error(
        _request: Request,
        exc: WaveformAccessError,
    ) -> JSONResponse:
        status_code = 404 if exc.code == "waveform_not_found" else 422
        if exc.code == "waveform_too_large":
            status_code = 413
        return _error_response(exc.code, str(exc), status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response("invalid_request", "request validation failed", 422)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_STATIC_ROOT / "index.html")

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        return {"schema_version": WEB_API_VERSION, "status": "ok"}

    @app.get("/api/v1/session")
    async def session() -> dict[str, Any]:
        return bridge.session()

    @app.get("/api/v1/devices")
    async def devices() -> dict[str, Any]:
        return await run_in_threadpool(bridge.devices)

    @app.get("/api/v1/waveforms")
    async def waveforms(directory: str = "", limit: int = 200) -> dict[str, Any]:
        entries = await run_in_threadpool(
            repository.list_directory,
            directory,
            limit=limit,
        )
        return {
            "schema_version": WEB_API_VERSION,
            "directory": directory,
            "entries": [entry.to_dict() for entry in entries],
        }

    @app.get("/api/v1/waveforms/preview")
    async def waveform_preview(path: str, points: int = 1024) -> dict[str, Any]:
        preview = await run_in_threadpool(repository.preview, path, points=points)
        return {"schema_version": WEB_API_VERSION, **preview}

    @app.post("/api/v1/commands", status_code=202)
    async def submit_command(request: Request) -> JSONResponse:
        payload = await _read_control_json(request)
        status = await run_in_threadpool(bridge.submit, payload)
        status_code = 202 if status["accepted"] else 409
        return JSONResponse(status, status_code=status_code)

    @app.get("/api/v1/commands/{command_id}")
    async def command_status(command_id: str) -> dict[str, Any]:
        return await run_in_threadpool(bridge.command_status, command_id)

    @app.post("/api/v1/stop")
    async def stop(request: Request) -> JSONResponse:
        payload = await _read_control_json(request)
        unknown = set(payload) - {"request_id"}
        if unknown:
            raise WebAPIError(
                "invalid_stop_request",
                f"unsupported stop fields: {sorted(unknown)}",
                status_code=422,
            )
        status = await run_in_threadpool(bridge.stop, payload.get("request_id"))
        return JSONResponse(status)

    @app.get("/api/v1/events")
    async def events(request: Request) -> StreamingResponse:
        if context.sse_clients >= MAX_SSE_CLIENTS:
            raise WebAPIError(
                "too_many_event_clients",
                "too many event-stream clients",
                status_code=429,
            )
        context.sse_clients += 1
        released = False

        def release_client() -> None:
            nonlocal released
            if released:
                return
            released = True
            context.sse_clients -= 1

        async def event_stream() -> AsyncIterator[str]:
            event_id = 0
            last_payload = ""
            heartbeat_deadline = asyncio.get_running_loop().time() + 15.0
            try:
                while not await request.is_disconnected():
                    encoded = await run_in_threadpool(context.event_state_json)
                    if encoded != last_payload:
                        event_id += 1
                        last_payload = encoded
                        yield f"id: {event_id}\nevent: state\ndata: {encoded}\n\n"
                        heartbeat_deadline = asyncio.get_running_loop().time() + 15.0
                    elif asyncio.get_running_loop().time() >= heartbeat_deadline:
                        yield ": keep-alive\n\n"
                        heartbeat_deadline = asyncio.get_running_loop().time() + 15.0
                    await asyncio.sleep(SSE_STATE_REFRESH_SECONDS)
            finally:
                release_client()

        return _GuardedEventStreamResponse(
            event_stream(),
            cleanup=release_client,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/v1/session/preview")
    async def current_preview(points: int = 1024) -> dict[str, Any]:
        return await run_in_threadpool(bridge.current_preview, points=points)

    @app.get("/api/v1/runs")
    async def runs(limit: int = 50) -> dict[str, Any]:
        return await run_in_threadpool(bridge.list_runs, limit=limit)

    @app.get("/api/v1/runs/{run_id}")
    async def run_detail(run_id: str, event_limit: int = 500) -> dict[str, Any]:
        return await run_in_threadpool(
            bridge.run_detail,
            run_id,
            event_limit=event_limit,
        )

    @app.get("/api/v1/runs/{run_id}/iterations/{iteration}/preview")
    async def iteration_preview(
        run_id: str,
        iteration: int,
        points: int = 1024,
    ) -> dict[str, Any]:
        return await run_in_threadpool(
            bridge.iteration_preview,
            run_id,
            iteration,
            points=points,
        )

    @app.get("/api/v1/runs/{run_id}/result.mat")
    async def run_result(run_id: str) -> StreamingResponse:
        handle, cleanup = await run_in_threadpool(
            _prepare_run_download,
            run_store,
            context.runs_directory,
            run_id,
        )
        try:
            return _GuardedDownloadResponse(
                handle,
                filename=f"result_{run_id}.mat",
                cleanup=cleanup,
            )
        except Exception:
            cleanup()
            raise

    @app.get("/api/v1/results/{command_id}.mat")
    async def command_result(command_id: str) -> StreamingResponse:
        handle, cleanup = await run_in_threadpool(
            _prepare_command_download,
            command_service,
            context.outbox_directory,
            command_id,
        )
        try:
            return _GuardedDownloadResponse(
                handle,
                filename=f"result_{command_id}.mat",
                cleanup=cleanup,
            )
        except Exception:
            cleanup()
            raise

    app.mount(
        "/static",
        StaticFiles(directory=_STATIC_ROOT, check_dir=True),
        name="static",
    )
    return app


def _prepare_run_download(
    run_store: RunStore,
    runs_directory: _AnchoredDirectory,
    run_id: str,
) -> tuple[BinaryIO, Callable[[], None]]:
    resources = ExitStack()
    try:
        try:
            recorder = resources.enter_context(run_store.export_guard(run_id))
        except (RunNotFoundError, TypeError, ValueError) as exc:
            raise WebAPIError(
                "run_not_found",
                "temporary run was not found",
                status_code=404,
            ) from exc

        try:
            if recorder.final_result_path is None:
                raise WebAPIError(
                    "result_not_available",
                    "completed result is not available",
                    status_code=409,
                )
            handle = runs_directory.open_file(run_id, "final_result.mat")
            resources.callback(handle.close)
            load_final_payload_file(handle)
            handle.seek(0)
        except WebAPIError:
            raise
        except Exception as exc:
            raise WebAPIError(
                "result_not_available",
                "completed result is unavailable or invalid",
                status_code=409,
            ) from exc
        return handle, resources.pop_all().close
    except Exception:
        resources.close()
        raise


def _prepare_command_download(
    command_service: FileCommandService,
    outbox_directory: _AnchoredDirectory,
    command_id: str,
) -> tuple[BinaryIO, Callable[[], None]]:
    try:
        result_name = command_service.result_path(command_id).name
    except FileCommandError as exc:
        raise WebAPIError(
            "result_not_found",
            "command result was not found",
            status_code=404,
        ) from exc

    resources = ExitStack()
    try:
        try:
            handle = outbox_directory.open_file(result_name)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WebAPIError(
                "result_not_found",
                "command result was not found",
                status_code=404,
            ) from exc
        resources.callback(handle.close)
        try:
            load_final_payload_file(handle)
            handle.seek(0)
        except (OSError, ResultExportError, ValueError) as exc:
            raise WebAPIError(
                "result_invalid",
                "command result is invalid",
                status_code=409,
            ) from exc
        return handle, resources.pop_all().close
    except Exception:
        resources.close()
        raise


async def _read_control_json(request: Request) -> dict[str, Any]:
    if request.headers.get(CONTROL_HEADER) != "1":
        raise WebAPIError(
            "control_header_required",
            f"{CONTROL_HEADER} header is required",
            status_code=403,
        )
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type != "application/json":
        raise WebAPIError(
            "json_content_type_required",
            "control requests require application/json",
            status_code=415,
        )
    encoding = request.headers.get("content-encoding", "").strip().lower()
    if encoding not in {"", "identity"}:
        raise WebAPIError(
            "content_encoding_not_supported",
            "compressed control requests are not supported",
            status_code=415,
        )
    _validate_origin(request)
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise WebAPIError(
                "invalid_content_length",
                "Content-Length must be an integer",
                status_code=400,
            ) from exc
        if declared_length > MAX_CONTROL_BODY_BYTES:
            raise WebAPIError(
                "request_too_large",
                "control request exceeds the body limit",
                status_code=413,
            )
    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_CONTROL_BODY_BYTES:
            raise WebAPIError(
                "request_too_large",
                "control request exceeds the body limit",
                status_code=413,
            )
        chunks.append(chunk)
    body = b"".join(chunks)
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WebAPIError(
            "invalid_json",
            "control request must use UTF-8 JSON",
            status_code=400,
        ) from exc
    try:
        payload = json.loads(
            text,
            parse_constant=lambda token: _reject_json_constant(token),
            object_pairs_hook=_unique_json_object,
        )
    except WebAPIError:
        raise
    except (RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WebAPIError(
            "invalid_json",
            "control request contains invalid JSON",
            status_code=400,
        ) from exc
    if not isinstance(payload, dict):
        raise WebAPIError(
            "invalid_json",
            "control request JSON must be an object",
            status_code=422,
        )
    depth, nodes = _json_shape(payload)
    if depth > MAX_JSON_DEPTH or nodes > MAX_JSON_NODES:
        raise WebAPIError(
            "json_too_complex",
            "control request JSON exceeds complexity limits",
            status_code=413,
        )
    return payload


def _validate_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is None:
        return
    if origin == "null":
        raise WebAPIError(
            "origin_not_allowed",
            "null Origin is not allowed",
            status_code=403,
        )
    parsed = urlsplit(origin)
    host = request.headers.get("host", "").lower()
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != host:
        raise WebAPIError(
            "origin_not_allowed",
            "control request Origin must match the console origin",
            status_code=403,
        )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WebAPIError(
                "duplicate_json_key",
                f"duplicate JSON key {key!r}",
                status_code=400,
            )
        result[key] = value
    return result


def _reject_json_constant(token: str) -> None:
    raise WebAPIError(
        "invalid_json_constant",
        f"invalid JSON constant {token}",
        status_code=400,
    )


def _json_shape(value: Any) -> tuple[int, int]:
    maximum_depth = 0
    nodes = 0
    stack = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        maximum_depth = max(maximum_depth, depth)
        if maximum_depth > MAX_JSON_DEPTH or nodes > MAX_JSON_NODES:
            return maximum_depth, nodes
        if isinstance(current, Mapping):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
    return maximum_depth, nodes


def _error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        {
            "schema_version": WEB_API_VERSION,
            "error": {"code": code, "message": message},
        },
        status_code=status_code,
    )


def _safe_path_component(value: object) -> bool:
    return (
        isinstance(value, str)
        and value not in {"", ".", ".."}
        and "/" not in value
        and "\\" not in value
        and "\x00" not in value
    )


def _required_flag(name: str) -> int:
    value = getattr(os, name, None)
    if value is None:
        raise RuntimeError(f"this platform does not provide required flag {name}")
    return int(value)


def _optional_flag(name: str) -> int:
    return int(getattr(os, name, 0))


__all__ = [
    "CONTROL_HEADER",
    "MAX_CONTROL_BODY_BYTES",
    "WEB_API_VERSION",
    "WebAPIError",
    "WebConsoleContext",
    "create_web_app",
]
