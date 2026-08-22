"""Deterministic waveform primitives for the reproducible PA experiments.

The waveform in this module is deliberately described as NR-like OFDM.  It
implements the numerology needed by the experiment protocol, but it does not
implement channel coding, a 3GPP resource mapper, or a standards receiver.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


ROOT_SEED = 20260822
DEFAULT_NFFT = 2048
DEFAULT_SYMBOL_COUNT = 16
DEFAULT_OCCUPIED_PER_SIDE = 600
DEFAULT_SAMPLE_COUNT = DEFAULT_NFFT * DEFAULT_SYMBOL_COUNT


def _name_to_spawn_key(parts: Iterable[object]) -> tuple[int, ...]:
    """Map a hierarchical name to an order-independent SeedSequence key."""

    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    # Four words are enough to make accidental collisions irrelevant while
    # keeping the spawn key easy to serialize in experiment manifests.
    return tuple(int(value) for value in np.frombuffer(digest[:16], dtype="<u4"))


def named_seed_sequence(*name: object) -> np.random.SeedSequence:
    """Return a deterministic named child of ``SeedSequence(20260822)``.

    ``SeedSequence.spawn`` is intentionally not used here because its output
    depends on call order.  A stable hash of the full hierarchical name is used
    as the child ``spawn_key`` instead, so parallel scheduling cannot change an
    experiment's random stream.
    """

    if not name or any(str(part) == "" for part in name):
        raise ValueError("a named seed requires at least one non-empty component")
    return np.random.SeedSequence(ROOT_SEED, spawn_key=_name_to_spawn_key(name))


def named_seed_sequences(namespace: str, count: int) -> tuple[np.random.SeedSequence, ...]:
    """Create ``count`` stable indexed children in a named namespace."""

    if not namespace:
        raise ValueError("namespace must not be empty")
    if count < 0:
        raise ValueError("count must be non-negative")
    return tuple(named_seed_sequence(namespace, index) for index in range(count))


def named_rng(*name: object) -> np.random.Generator:
    """Create a NumPy generator backed by a deterministic named child seed."""

    return np.random.default_rng(named_seed_sequence(*name))


@dataclass(frozen=True)
class OFDMWaveformConfig:
    """Configuration for the frozen NR-like OFDM waveform."""

    nfft: int = DEFAULT_NFFT
    symbol_count: int = DEFAULT_SYMBOL_COUNT
    occupied_per_side: int = DEFAULT_OCCUPIED_PER_SIDE
    qam_order: int = 256
    cyclic_prefix_length: int = 0

    def __post_init__(self) -> None:
        if self.nfft <= 0 or self.symbol_count <= 0:
            raise ValueError("nfft and symbol_count must be positive")
        if self.occupied_per_side <= 0:
            raise ValueError("occupied_per_side must be positive")
        if 2 * self.occupied_per_side + 1 > self.nfft:
            raise ValueError("occupied subcarriers and DC do not fit in nfft")
        if self.qam_order != 256:
            raise ValueError("the frozen experiment waveform uses 256-QAM")
        if self.cyclic_prefix_length != 0:
            raise ValueError("cyclic prefix is disabled for periodic ILC")

    @property
    def sample_count(self) -> int:
        return self.nfft * self.symbol_count

    @property
    def occupied_subcarrier_count(self) -> int:
        return 2 * self.occupied_per_side


@dataclass(frozen=True)
class OFDMWaveform:
    """A time-domain waveform and its exactly matching known resource grid."""

    samples: NDArray[np.complex128]
    grid: NDArray[np.complex128]
    occupied_bins: NDArray[np.int64]
    config: OFDMWaveformConfig
    seed_spawn_key: tuple[int, ...]

    @property
    def time_domain(self) -> NDArray[np.complex128]:
        return self.samples

    @property
    def frequency_grid(self) -> NDArray[np.complex128]:
        return self.grid


def occupied_subcarrier_bins(
    nfft: int = DEFAULT_NFFT,
    occupied_per_side: int = DEFAULT_OCCUPIED_PER_SIDE,
) -> NDArray[np.int64]:
    """Return FFT-order indices for bins ``-K..-1`` and ``1..K``."""

    if nfft <= 0 or occupied_per_side <= 0:
        raise ValueError("nfft and occupied_per_side must be positive")
    if 2 * occupied_per_side + 1 > nfft:
        raise ValueError("occupied bins and DC do not fit in nfft")
    positive = np.arange(1, occupied_per_side + 1, dtype=np.int64)
    negative = np.arange(nfft - occupied_per_side, nfft, dtype=np.int64)
    return np.concatenate((negative, positive))


def qam256_symbols(
    rng: np.random.Generator,
    shape: int | tuple[int, ...],
) -> NDArray[np.complex128]:
    """Draw Gray-label-independent, unit-average-energy square 256-QAM.

    Only the constellation point distribution is relevant to these waveform
    experiments.  Coding and bit labels are intentionally outside scope.
    """

    real_index = rng.integers(0, 16, size=shape, dtype=np.int64)
    imag_index = rng.integers(0, 16, size=shape, dtype=np.int64)
    real_level = 2.0 * real_index - 15.0
    imag_level = 2.0 * imag_index - 15.0
    # Each 16-PAM dimension has mean energy 85, hence 170 for complex QAM.
    return np.asarray((real_level + 1j * imag_level) / np.sqrt(170.0), dtype=np.complex128)


def _coerce_seed_sequence(
    seed: np.random.SeedSequence | str | int | tuple[object, ...] | None,
) -> np.random.SeedSequence:
    if seed is None:
        return named_seed_sequence("waveform", "default")
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if isinstance(seed, tuple):
        return named_seed_sequence("waveform", *seed)
    return named_seed_sequence("waveform", seed)


def generate_ofdm_waveform(
    seed: np.random.SeedSequence | str | int | tuple[object, ...] | None = None,
    config: OFDMWaveformConfig | None = None,
) -> OFDMWaveform:
    """Generate the frozen deterministic, unit-power NR-like OFDM waveform.

    The 2048-point IFFT uses bins ``-600..-1`` and ``1..600``.  DC and all
    remaining guard bins are exactly zero.  Sixteen symbols are concatenated
    without a cyclic prefix, producing exactly 32768 complex samples.
    """

    cfg = config if config is not None else OFDMWaveformConfig()
    seed_sequence = _coerce_seed_sequence(seed)
    rng = np.random.default_rng(seed_sequence)
    bins = occupied_subcarrier_bins(cfg.nfft, cfg.occupied_per_side)

    grid = np.zeros((cfg.symbol_count, cfg.nfft), dtype=np.complex128)
    grid[:, bins] = qam256_symbols(
        rng,
        (cfg.symbol_count, cfg.occupied_subcarrier_count),
    )

    symbols = np.fft.ifft(grid, axis=1, norm="ortho")
    rms = float(np.sqrt(np.mean(np.abs(symbols) ** 2)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise RuntimeError("generated waveform has invalid power")

    # Apply the same scalar to the grid and samples so a subsequent unitary FFT
    # recovers the exact known grid used by the EVM metric.
    grid /= rms
    symbols /= rms
    samples = np.ascontiguousarray(symbols.reshape(-1), dtype=np.complex128)
    return OFDMWaveform(
        samples=samples,
        grid=np.ascontiguousarray(grid),
        occupied_bins=bins,
        config=cfg,
        seed_spawn_key=tuple(seed_sequence.spawn_key),
    )


def normalize_unit_power(samples: ArrayLike) -> NDArray[np.complex128]:
    """Return a complex128 copy normalized to unit mean power."""

    values = np.asarray(samples, dtype=np.complex128)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be non-empty and finite")
    rms = float(np.sqrt(np.mean(np.abs(values) ** 2)))
    if rms <= 0.0:
        raise ValueError("zero-power samples cannot be normalized")
    return np.asarray(values / rms, dtype=np.complex128)


__all__ = [
    "DEFAULT_NFFT",
    "DEFAULT_OCCUPIED_PER_SIDE",
    "DEFAULT_SAMPLE_COUNT",
    "DEFAULT_SYMBOL_COUNT",
    "OFDMWaveform",
    "OFDMWaveformConfig",
    "ROOT_SEED",
    "generate_ofdm_waveform",
    "named_rng",
    "named_seed_sequence",
    "named_seed_sequences",
    "normalize_unit_power",
    "occupied_subcarrier_bins",
    "qam256_symbols",
]
