"""Generic MATLAB file loading and atomic saving helpers."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import MatProtocolError, UnsupportedMatVersion


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
    try:
        from scipy.io import loadmat

        raw = loadmat(path, squeeze_me=True, struct_as_record=False)
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
                f"{path} is likely MATLAB v7.3; install h5py to read it"
            ) from h5_exc
        with h5py.File(path, "r") as handle:
            return {key: _h5_value(value) for key, value in handle.items()}
    except (OSError, ValueError) as exc:
        raise MatProtocolError(f"failed to read MAT file {path}: {exc}") from exc


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
        tmp.replace(path)
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise MatProtocolError(f"failed to write MAT file {path}: {exc}") from exc
