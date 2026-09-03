"""Generic MATLAB file loading and atomic saving helpers."""

from __future__ import annotations

import os
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .exceptions import MatProtocolError, UnsupportedMatVersion

_REPLACE_RETRY_ATTEMPTS = 10
_REPLACE_RETRY_DELAY_SECONDS = 0.02


def replace_with_retry(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
    """Atomically replace ``destination`` with a short Windows retry loop.

    On Windows, antivirus or indexing services may briefly hold a freshly
    written file, which makes an immediate ``os.replace`` fail with
    PermissionError even though this process keeps no handle open. Retry the
    replacement for a bounded window on Windows only; POSIX keeps the
    single-shot semantics because a PermissionError there is a real
    permission problem.
    """

    if sys.platform != "win32":
        os.replace(source, destination)
        return
    for attempt in range(_REPLACE_RETRY_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_RETRY_DELAY_SECONDS)


def _unwrap(value: Any) -> Any:
    """Convert scipy MATLAB proxy objects into plain Python containers."""
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            if value.size == 1:
                value = value.reshape(-1)[0]
            return {name: _unwrap(value[name]) for name in value.dtype.names}
        if value.dtype == object:
            if value.size == 1:
                return _unwrap(value.reshape(-1)[0])
            return [_unwrap(item) for item in value.reshape(-1)]
        return value
    if hasattr(value, "_fieldnames"):
        return {name: _unwrap(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, Mapping):
        return {str(key): _unwrap(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_unwrap(item) for item in value]
    return value


def load_mat(path: str | Path) -> dict[str, Any]:
    """Load a MAT file without requiring MATLAB.

    SciPy handles MAT v5/v6/v7 files. MAT v7.3 is HDF5 and is supported
    opportunistically when h5py is installed.
    """
    path = Path(path)
    return _load_mat_source(path, label=str(path))


def load_mat_file(handle: BinaryIO) -> dict[str, Any]:
    """Load a MAT payload from an already opened seekable binary file."""
    if not hasattr(handle, "read") or not hasattr(handle, "seek"):
        raise TypeError("handle must be a seekable binary file")
    try:
        handle.seek(0)
    except (OSError, ValueError) as exc:
        raise MatProtocolError("failed to seek opened MAT file") from exc
    return _load_mat_source(handle, label="opened MAT file")


def _load_mat_source(source: str | Path | BinaryIO, *, label: str) -> dict[str, Any]:
    """Decode one path or caller-owned file object without closing it."""
    try:
        from scipy.io import loadmat

        raw = loadmat(source, squeeze_me=True, struct_as_record=False)
        return {
            key: _unwrap(value)
            for key, value in raw.items()
            if not key.startswith("__")
        }
    except NotImplementedError:
        try:
            import h5py  # type: ignore
        except ImportError as h5_exc:
            raise UnsupportedMatVersion(
                f"{label} is likely MATLAB v7.3; install h5py to read it"
            ) from h5_exc
        if hasattr(source, "seek"):
            try:
                source.seek(0)  # type: ignore[union-attr]
            except (OSError, ValueError) as exc:
                raise MatProtocolError(f"failed to seek {label}") from exc
        with h5py.File(source, "r") as handle:
            return {key: _h5_value(value) for key, value in handle.items()}
    except (OSError, ValueError) as exc:
        raise MatProtocolError(f"failed to read MAT file {label}: {exc}") from exc


def _h5_value(value: Any) -> Any:
    data = value[()]
    if hasattr(data, "dtype") and data.dtype.kind in "fiu":
        return np.asarray(data)
    if hasattr(data, "dtype") and data.dtype.kind == "c":
        return np.asarray(data)
    return data


def save_mat(path: str | Path, values: Mapping[str, Any]) -> None:
    """Write a MATLAB-compatible v5 MAT file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each writer needs its own temporary path. Concurrent replacements of the
    # same destination are safe and follow last-completed-writer-wins semantics.
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.mat")
    try:
        from scipy.io import savemat

        savemat(
            tmp,
            dict(values),
            appendmat=False,
            do_compression=False,
            long_field_names=True,
        )
        replace_with_retry(tmp, path)
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise MatProtocolError(f"failed to write MAT file {path}: {exc}") from exc
