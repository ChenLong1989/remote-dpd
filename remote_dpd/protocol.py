"""Compatibility boundary for the legacy remote DPD MAT file protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .exceptions import MatProtocolError, UnsupportedMatVersion

CONFIG_FILE = "Config_file.mat"
DPD_IN_FILE = "DPD_in.mat"
FB_SIGNAL_FILE = "FB_Signal.mat"
CONFIG_ACK_FILE = "Config_file_ack.mat"
DPD_IN_ACK_FILE = "ACK_DPDin.mat"
DPD_OUT_FILE = "DPDout_Nokia.mat"
SYMBOL_EVM_FILE = "symbolEVM.mat"
HEARTBEAT_FILE = "sync_dat.txt"


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


def _mat_candidates(path: Path) -> list[Path]:
    if path.suffix.lower() == ".mat":
        return [path]
    return [path, path.with_suffix(".mat")]


def resolve_file(directory: str | Path, name: str) -> Path:
    """Resolve both MATLAB's extensionless and normal `.mat` spellings."""
    directory = Path(directory)
    for candidate in _mat_candidates(directory / name):
        if candidate.is_file():
            return candidate
    return directory / (name if name.endswith(".mat") else f"{name}.mat")


def load_mat(path: str | Path) -> dict[str, Any]:
    """Load a MAT file without requiring MATLAB.

    scipy handles the v5/v6/v7 formats used by the legacy exchange. v7.3 is
    HDF5 and is supported opportunistically when h5py is installed.
    """
    path = Path(path)
    try:
        from scipy.io import loadmat

        raw = loadmat(path, squeeze_me=True, struct_as_record=False)
        return {key: _unwrap(value) for key, value in raw.items() if not key.startswith("__")}
    except NotImplementedError as exc:
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
    # Keep the temporary filename's `.mat` suffix: scipy otherwise appends a
    # second `.mat` and the atomic replace would target a nonexistent path.
    tmp = path.with_name(f".{path.stem}.tmp.mat")
    try:
        from scipy.io import savemat

        savemat(tmp, dict(values), do_compression=False, long_field_names=True)
        tmp.replace(path)
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise MatProtocolError(f"failed to write MAT file {path}: {exc}") from exc


def as_vector(value: Any, variable: str, *, dtype=np.complex128) -> np.ndarray:
    """Validate and normalize a legacy IQ variable to a one-dimensional vector."""
    if value is None:
        raise MatProtocolError(f"MAT file is missing {variable}")
    array = np.asarray(value)
    if array.size == 0:
        raise MatProtocolError(f"{variable} is empty")
    if not np.issubdtype(array.dtype, np.number):
        raise MatProtocolError(f"{variable} must be numeric, got {array.dtype}")
    return np.asarray(array, dtype=dtype).reshape(-1)


def first_value(payload: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return default
