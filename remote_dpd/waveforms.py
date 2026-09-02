"""Symlink-safe local MAT waveform browsing and bounded preview helpers."""

from __future__ import annotations

import io
import math
import os
import stat
import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .safety import check_reference

DEFAULT_MAX_WAVEFORM_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_WAVEFORM_SAMPLES = 10_000_000
MAX_PREVIEW_POINTS = 4_096

#: Windows cannot open directory descriptors at all, so the fd-anchored
#: access model runs on POSIX only. On Windows the repository falls back to
#: path-based access that re-validates every component with lstat (rejecting
#: symlinks and any other reparse point) before each open; the check-then-use
#: window is accepted for the trusted single-user console threat model.
DIRECTORY_FDS_SUPPORTED = sys.platform != "win32"


class WaveformAccessError(ValueError):
    """A waveform path or MAT payload is unsafe or invalid."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class WaveformEntry:
    """One safe direct child returned by a waveform-directory listing."""

    path: str
    name: str
    kind: str
    size_bytes: int | None
    modified_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "modified_ns": self.modified_ns,
        }


class WaveformRepository:
    """Keep an anchored directory descriptor for all waveform operations."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_bytes: int = DEFAULT_MAX_WAVEFORM_BYTES,
        max_samples: int = DEFAULT_MAX_WAVEFORM_SAMPLES,
    ) -> None:
        self._max_bytes = _positive_integer(max_bytes, "max_bytes")
        self._max_samples = _positive_integer(max_samples, "max_samples")
        raw_root = Path(root).expanduser()
        raw_root.mkdir(parents=True, exist_ok=True)
        if raw_root.is_symlink() or not raw_root.is_dir():
            raise WaveformAccessError(
                "unsafe_waveform_root",
                "waveform root must be a real directory",
            )
        self._root = raw_root.resolve(strict=True)
        self._root_fd: int | None = None
        if DIRECTORY_FDS_SUPPORTED:
            flags = os.O_RDONLY | _flag("O_DIRECTORY") | _flag("O_CLOEXEC")
            flags |= _flag("O_NOFOLLOW")
            try:
                self._root_fd = os.open(self._root, flags)
            except OSError as exc:
                raise WaveformAccessError(
                    "unsafe_waveform_root",
                    "waveform root cannot be opened safely",
                ) from exc
        self._closed = False

    @property
    def root(self) -> Path:
        """Return the canonical root for diagnostics, never for API responses."""
        return self._root

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._root_fd is not None:
            os.close(self._root_fd)

    def list_directory(
        self,
        relative_directory: str = "",
        *,
        limit: int = 200,
    ) -> tuple[WaveformEntry, ...]:
        """List safe direct children without following symbolic links."""
        normalized_limit = _bounded_integer(limit, "limit", minimum=1, maximum=500)
        components = _relative_components(relative_directory, allow_empty=True)
        parent = "/".join(components)
        if DIRECTORY_FDS_SUPPORTED:
            directory_fd = self._open_directory(components)
            try:
                with os.scandir(directory_fd) as iterator:
                    return self._collect_entries(iterator, parent, normalized_limit)
            finally:
                os.close(directory_fd)
        directory = self._anchored_directory_path(components)
        with os.scandir(directory) as iterator:
            return self._collect_entries(iterator, parent, normalized_limit)

    def _collect_entries(
        self,
        iterator: Iterator[os.DirEntry],
        parent: str,
        limit: int,
    ) -> tuple[WaveformEntry, ...]:
        entries: list[WaveformEntry] = []
        for entry in iterator:
            if entry.name.startswith(".") or entry.is_symlink():
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if _is_linked_metadata(metadata):
                continue
            relative = entry.name if not parent else f"{parent}/{entry.name}"
            if stat.S_ISDIR(metadata.st_mode):
                entries.append(
                    WaveformEntry(
                        relative,
                        entry.name,
                        "directory",
                        None,
                        metadata.st_mtime_ns,
                    )
                )
            elif stat.S_ISREG(metadata.st_mode) and entry.name.lower().endswith(
                ".mat"
            ):
                entries.append(
                    WaveformEntry(
                        relative,
                        entry.name,
                        "waveform",
                        metadata.st_size,
                        metadata.st_mtime_ns,
                    )
                )
        entries.sort(key=lambda item: (item.kind != "directory", item.name.lower()))
        return tuple(entries[:limit])

    def load_x(self, relative_path: str) -> np.ndarray:
        """Open, read, and validate x from one anchored regular MAT file."""
        components = _relative_components(relative_path, allow_empty=False)
        if not components[-1].lower().endswith(".mat"):
            raise WaveformAccessError(
                "invalid_waveform_path",
                "waveform file must use the .mat extension",
            )
        if DIRECTORY_FDS_SUPPORTED:
            payload = self._read_file_bytes_posix(components)
        else:
            payload = self._read_file_bytes_windows(components)

        raw_x = _load_x_from_mat_bytes(payload, max_samples=self._max_samples)
        array = np.asarray(raw_x)
        if array.size == 1:
            raise WaveformAccessError(
                "invalid_waveform",
                "x must contain more than one sample",
            )
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1 or array.size == 0:
            raise WaveformAccessError(
                "invalid_waveform",
                "x must be a non-empty row, column, or one-dimensional vector",
            )
        if array.size > self._max_samples:
            raise WaveformAccessError(
                "waveform_too_large",
                f"x exceeds {self._max_samples} samples",
            )
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype,
            np.bool_,
        ):
            raise WaveformAccessError("invalid_waveform", "x must be numeric IQ data")
        try:
            copied = np.array(array, dtype=np.complex128, order="C", copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise WaveformAccessError(
                "invalid_waveform",
                "x cannot be converted to complex128",
            ) from exc
        if not np.all(np.isfinite(copied)):
            raise WaveformAccessError(
                "invalid_waveform",
                "x must contain only finite samples",
            )
        report = check_reference(copied)
        if report.reference_rms == 0.0:
            raise WaveformAccessError(
                "unsafe_waveform",
                "x must have non-zero RMS power",
            )
        copied.setflags(write=False)
        return copied

    def preview(self, relative_path: str, *, points: int = 1024) -> dict[str, Any]:
        """Return bounded samples and digital safety metadata for x."""
        normalized_points = _bounded_integer(
            points,
            "points",
            minimum=16,
            maximum=MAX_PREVIEW_POINTS,
        )
        x = self.load_x(relative_path)
        indices = _sample_indices(x.size, normalized_points)
        selected = x[indices]
        report = check_reference(x)
        return {
            "path": relative_path,
            "sample_count": int(x.size),
            "preview_count": int(indices.size),
            "index": indices.tolist(),
            "real": selected.real.tolist(),
            "imag": selected.imag.tolist(),
            "magnitude": np.abs(selected).tolist(),
            "safety": report.to_dict(),
        }

    def _read_file_bytes_posix(self, components: tuple[str, ...]) -> bytes:
        parent_fd = self._open_directory(components[:-1])
        file_fd: int | None = None
        try:
            flags = os.O_RDONLY | _flag("O_CLOEXEC") | _flag("O_NOFOLLOW")
            flags |= _flag("O_NONBLOCK")
            try:
                file_fd = os.open(components[-1], flags, dir_fd=parent_fd)
            except FileNotFoundError as exc:
                raise WaveformAccessError(
                    "waveform_not_found",
                    "waveform file was not found",
                ) from exc
            except OSError as exc:
                raise WaveformAccessError(
                    "unsafe_waveform_path",
                    "waveform file cannot be opened safely",
                ) from exc
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise WaveformAccessError(
                    "unsafe_waveform_path",
                    "waveform path must identify a regular file",
                )
            if metadata.st_size > self._max_bytes:
                raise WaveformAccessError(
                    "waveform_too_large",
                    f"waveform file exceeds {self._max_bytes} bytes",
                )
            with os.fdopen(file_fd, "rb", closefd=True) as handle:
                file_fd = None
                payload = handle.read(self._max_bytes + 1)
            if len(payload) > self._max_bytes:
                raise WaveformAccessError(
                    "waveform_too_large",
                    f"waveform file exceeds {self._max_bytes} bytes",
                )
            return payload
        finally:
            if file_fd is not None:
                os.close(file_fd)
            os.close(parent_fd)

    def _read_file_bytes_windows(self, components: tuple[str, ...]) -> bytes:
        directory = self._anchored_directory_path(components[:-1])
        target = directory / components[-1]
        try:
            metadata = os.lstat(target)
        except FileNotFoundError as exc:
            raise WaveformAccessError(
                "waveform_not_found",
                "waveform file was not found",
            ) from exc
        except OSError as exc:
            raise WaveformAccessError(
                "unsafe_waveform_path",
                "waveform file cannot be opened safely",
            ) from exc
        _reject_linked_metadata(metadata)
        if not stat.S_ISREG(metadata.st_mode):
            raise WaveformAccessError(
                "unsafe_waveform_path",
                "waveform path must identify a regular file",
            )
        if metadata.st_size > self._max_bytes:
            raise WaveformAccessError(
                "waveform_too_large",
                f"waveform file exceeds {self._max_bytes} bytes",
            )
        file_fd: int | None = None
        try:
            file_fd = os.open(target, os.O_RDONLY | _flag("O_BINARY"))
            opened = os.fstat(file_fd)
            if not stat.S_ISREG(opened.st_mode):
                raise WaveformAccessError(
                    "unsafe_waveform_path",
                    "waveform path must identify a regular file",
                )
            if opened.st_size > self._max_bytes:
                raise WaveformAccessError(
                    "waveform_too_large",
                    f"waveform file exceeds {self._max_bytes} bytes",
                )
            with os.fdopen(file_fd, "rb", closefd=True) as handle:
                file_fd = None
                payload = handle.read(self._max_bytes + 1)
            if len(payload) > self._max_bytes:
                raise WaveformAccessError(
                    "waveform_too_large",
                    f"waveform file exceeds {self._max_bytes} bytes",
                )
            return payload
        except WaveformAccessError:
            raise
        except OSError as exc:
            raise WaveformAccessError(
                "unsafe_waveform_path",
                "waveform file cannot be opened safely",
            ) from exc
        finally:
            if file_fd is not None:
                os.close(file_fd)

    def _anchored_directory_path(self, components: tuple[str, ...]) -> Path:
        """Resolve components under the pinned root, rejecting linked paths.

        Windows fallback for the fd-anchored POSIX model: every component is
        re-validated with lstat (symlinks and any reparse point are refused)
        before the resulting path is handed to the caller.
        """

        self._require_open()
        current = self._root
        try:
            for component in components:
                child = current / component
                try:
                    metadata = os.lstat(child)
                except FileNotFoundError as exc:
                    raise WaveformAccessError(
                        "waveform_not_found",
                        "waveform directory was not found",
                    ) from exc
                _reject_linked_metadata(metadata)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise WaveformAccessError(
                        "unsafe_waveform_path",
                        "waveform path must identify a directory",
                    )
                current = child
            return current
        except OSError as exc:
            raise WaveformAccessError(
                "unsafe_waveform_path",
                "waveform directory cannot be opened safely",
            ) from exc

    def _open_directory(self, components: tuple[str, ...]) -> int:
        self._require_open()
        assert self._root_fd is not None  # POSIX-only call site
        current_fd = os.dup(self._root_fd)
        try:
            flags = os.O_RDONLY | _flag("O_DIRECTORY") | _flag("O_CLOEXEC")
            flags |= _flag("O_NOFOLLOW")
            for component in components:
                next_fd = os.open(component, flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            return current_fd
        except FileNotFoundError as exc:
            os.close(current_fd)
            raise WaveformAccessError(
                "waveform_not_found",
                "waveform directory was not found",
            ) from exc
        except OSError as exc:
            os.close(current_fd)
            raise WaveformAccessError(
                "unsafe_waveform_path",
                "waveform directory cannot be opened safely",
            ) from exc

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("waveform repository is closed")


def _load_x_from_mat_bytes(payload: bytes, *, max_samples: int) -> Any:
    try:
        from scipy.io import loadmat, whosmat
        from scipy.io.matlab import MatReadError

        variables = {
            name: (shape, value_class)
            for name, shape, value_class in whosmat(io.BytesIO(payload))
        }
        if "x" not in variables:
            raise WaveformAccessError(
                "waveform_variable_missing",
                "MAT file does not contain variable x",
            )
        shape, value_class = variables["x"]
        if value_class not in {
            "double",
            "single",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        }:
            raise WaveformAccessError(
                "invalid_waveform",
                f"x must not use MATLAB class {value_class}",
            )
        _validate_waveform_shape(shape, max_samples=max_samples)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The default value for `spmatrix` is changing.*",
                category=DeprecationWarning,
            )
            raw = loadmat(
                io.BytesIO(payload),
                squeeze_me=False,
                struct_as_record=False,
                variable_names=["x"],
            )
        value = raw["x"]
        try:
            from scipy.sparse import issparse

            if issparse(value):
                raise WaveformAccessError(
                    "invalid_waveform",
                    "x must not be a sparse array",
                )
        except ImportError:  # pragma: no cover - scipy.io already requires scipy
            pass
        return value
    except WaveformAccessError:
        raise
    except NotImplementedError:
        return _load_hdf5_x(payload, max_samples=max_samples)
    except (MatReadError, OSError, TypeError, ValueError) as exc:
        raise WaveformAccessError(
            "invalid_mat",
            "waveform MAT file cannot be decoded",
        ) from exc


def _load_hdf5_x(payload: bytes, *, max_samples: int) -> Any:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise WaveformAccessError(
            "unsupported_mat_version",
            "MAT v7.3 waveform requires the optional h5py package",
        ) from exc
    try:
        with h5py.File(io.BytesIO(payload), "r") as handle:
            link = handle.get("x", getlink=True)
            if link is None:
                raise WaveformAccessError(
                    "waveform_variable_missing",
                    "MAT file does not contain variable x",
                )
            if not isinstance(link, h5py.HardLink):
                raise WaveformAccessError(
                    "unsafe_waveform",
                    "MAT v7.3 x must not use an external or symbolic link",
                )
            dataset = handle["x"]
            if not isinstance(dataset, h5py.Dataset):
                raise WaveformAccessError(
                    "invalid_waveform",
                    "MAT v7.3 x must be a dataset",
                )
            if dataset.is_virtual or dataset.external:
                raise WaveformAccessError(
                    "unsafe_waveform",
                    "MAT v7.3 x must not use virtual or external storage",
                )
            _validate_hdf5_dataset_layout(
                shape=dataset.shape,
                dtype=dataset.dtype,
                maxshape=dataset.maxshape,
                chunks=dataset.chunks,
                max_samples=max_samples,
            )
            return np.asarray(dataset)
    except WaveformAccessError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise WaveformAccessError(
            "invalid_mat",
            "MAT v7.3 waveform cannot be decoded",
        ) from exc


def _validate_waveform_shape(shape: object, *, max_samples: int) -> int:
    try:
        dimensions = tuple(int(item) for item in shape)  # type: ignore[union-attr]
    except (TypeError, ValueError, OverflowError) as exc:
        raise WaveformAccessError(
            "invalid_waveform",
            "x has an invalid array shape",
        ) from exc
    if not dimensions or any(item < 0 for item in dimensions):
        raise WaveformAccessError(
            "invalid_waveform",
            "x has an invalid array shape",
        )
    if len(dimensions) > 2 or (
        len(dimensions) == 2 and dimensions[0] != 1 and dimensions[1] != 1
    ):
        raise WaveformAccessError(
            "invalid_waveform",
            "x must be a row, column, or one-dimensional vector",
        )
    sample_count = 1
    for dimension in dimensions:
        if dimension == 0:
            sample_count = 0
            break
        if sample_count > max_samples // dimension:
            raise WaveformAccessError(
                "waveform_too_large",
                f"x exceeds {max_samples} samples",
            )
        sample_count *= dimension
    if sample_count <= 1:
        raise WaveformAccessError(
            "invalid_waveform",
            "x must contain more than one sample",
        )
    return sample_count


def _validate_hdf5_dataset_layout(
    *,
    shape: object,
    dtype: object,
    maxshape: object,
    chunks: object,
    max_samples: int,
) -> None:
    sample_count = _validate_waveform_shape(shape, max_samples=max_samples)
    normalized_dtype = np.dtype(dtype)
    if (
        normalized_dtype.kind not in "iufc"
        or normalized_dtype.fields is not None
        or normalized_dtype.subdtype is not None
        or (normalized_dtype.kind in "iu" and normalized_dtype.itemsize > 8)
        or (normalized_dtype.kind == "f" and normalized_dtype.itemsize > 8)
        or (normalized_dtype.kind == "c" and normalized_dtype.itemsize > 16)
    ):
        raise WaveformAccessError(
            "invalid_waveform",
            "MAT v7.3 x must use a regular numeric dtype",
        )
    try:
        fixed_shape = tuple(int(item) for item in maxshape)  # type: ignore[union-attr]
        current_shape = tuple(int(item) for item in shape)  # type: ignore[union-attr]
    except (TypeError, ValueError, OverflowError) as exc:
        raise WaveformAccessError(
            "unsafe_waveform",
            "MAT v7.3 x must use a fixed dataset shape",
        ) from exc
    if fixed_shape != current_shape:
        raise WaveformAccessError(
            "unsafe_waveform",
            "MAT v7.3 x must use a fixed dataset shape",
        )
    if chunks is None:
        return
    try:
        chunk_elements = math.prod(int(item) for item in chunks)  # type: ignore[union-attr]
    except (TypeError, ValueError, OverflowError) as exc:
        raise WaveformAccessError(
            "unsafe_waveform",
            "MAT v7.3 x uses an invalid storage chunk",
        ) from exc
    if (
        chunk_elements <= 0
        or chunk_elements > sample_count
        or chunk_elements > max_samples
        or chunk_elements * normalized_dtype.itemsize > max_samples * 16
    ):
        raise WaveformAccessError(
            "waveform_too_large",
            "MAT v7.3 x uses an oversized storage chunk",
        )


def _relative_components(value: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise WaveformAccessError(
            "invalid_waveform_path",
            "waveform path must be a string",
        )
    if "\x00" in value or "\\" in value or ":" in value:
        raise WaveformAccessError(
            "invalid_waveform_path",
            "waveform path contains unsupported characters",
        )
    if value == "":
        if allow_empty:
            return ()
        raise WaveformAccessError(
            "invalid_waveform_path",
            "waveform path must not be empty",
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(item in {"", ".", ".."} for item in path.parts)
    ):
        raise WaveformAccessError(
            "invalid_waveform_path",
            "waveform path must be a normalized relative POSIX path",
        )
    return tuple(path.parts)


def _sample_indices(sample_count: int, points: int) -> np.ndarray:
    if sample_count <= points:
        return np.arange(sample_count, dtype=np.int64)
    return np.unique(np.linspace(0, sample_count - 1, points, dtype=np.int64))


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _bounded_integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    normalized = _positive_integer(value, name)
    if normalized < minimum or normalized > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return normalized


def _flag(name: str) -> int:
    return int(getattr(os, name, 0))


def _is_linked_metadata(metadata: os.stat_result) -> bool:
    """True when the lstat metadata describes a symlink or reparse point."""

    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_reparse_tag", 0)
    )


def _reject_linked_metadata(metadata: os.stat_result) -> None:
    if _is_linked_metadata(metadata):
        raise WaveformAccessError(
            "unsafe_waveform_path",
            "waveform path must not contain symbolic links or reparse points",
        )


__all__ = [
    "DEFAULT_MAX_WAVEFORM_BYTES",
    "DEFAULT_MAX_WAVEFORM_SAMPLES",
    "MAX_PREVIEW_POINTS",
    "WaveformAccessError",
    "WaveformEntry",
    "WaveformRepository",
]
