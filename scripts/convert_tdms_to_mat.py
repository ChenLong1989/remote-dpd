"""TEMPORARY waveform conversion tool (transition measure, not a product feature).

The project will later provide its own waveform-source generator; until that
tool exists, this script converts the station GUI's TDMS waveform (for example
NR_DL_TM1.1.tdms, 5G NR 100 MHz TM1.1, NI RFmx "InterleavedIQCluster" layout)
into the project MAT contract: one complex column vector named ``x`` sampled
at the closed-loop rate (491.52 MS/s by default).

Processing steps, all explicit and amplitude preserving:

1. De-interleave the single TDMS float32 stream into I/Q pairs.
2. Rationally resample from the TDMS ``NI_RF_IQRate`` to the target closed-loop
   rate (122.88 MS/s -> 491.52 MS/s is an exact 4x upsample).
3. Cut a leading span of the requested length (default one 0.5 ms NR slot =
   245,760 samples at 491.52 MS/s) starting at a slot boundary.

No scaling, normalization, or runtime gain is applied; the NI RFmx
``NI_RF_RuntimeScaling`` value is reported but not folded into the samples.

Usage:
    python scripts/convert_tdms_to_mat.py <input.tdms> <output.mat> \
        [--length-samples N] [--offset-samples N] [--target-rate HZ] [--info]

Requires the throwaway dependency ``nptdms`` (pip install nptdms).
Remove this script once the dedicated waveform-source tool ships.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_LENGTH_SAMPLES = 245_760  # one 0.5 ms NR slot at 491.52 MS/s
DEFAULT_TARGET_RATE_HZ = 491.52e6


def load_tdms_iq(path: Path):
    """Return (iq complex array, tdms_iq_rate_hz, metadata) from a TDMS file."""

    import numpy as np
    from nptdms import TdmsFile

    tdms = TdmsFile.read(path)
    channels = [c for group in tdms.groups() for c in group.channels()]
    if not channels:
        raise SystemExit(f"no channels found in {path}")

    channel = channels[0]
    properties = dict(channel.properties)
    waveform_type = str(properties.get("NI_RF_WaveformType", ""))
    iq_rate = float(properties.get("NI_RF_IQRate", 0.0))
    if iq_rate <= 0.0:
        raise SystemExit("TDMS channel lacks a valid NI_RF_IQRate property")

    raw = np.asarray(channel[:])
    if waveform_type == "InterleavedIQCluster" or (
        raw.ndim == 1 and raw.size % 2 == 0 and not np.iscomplexobj(raw)
    ):
        if raw.size % 2 != 0:
            raise SystemExit("interleaved IQ stream has an odd sample count")
        iq = raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)
    elif np.iscomplexobj(raw):
        iq = np.asarray(raw, dtype=np.complex128)
    else:
        raise SystemExit(
            f"unsupported TDMS waveform layout {waveform_type!r}; inspect with --info"
        )
    return iq, iq_rate, properties


def resample_to_rate(iq, source_rate_hz: float, target_rate_hz: float):
    """Rationally resample ``iq`` from source to target rate (gain of one)."""

    import math

    from scipy.signal import resample_poly

    factor = target_rate_hz / source_rate_hz
    if math.isclose(factor, 1.0, rel_tol=1e-9):
        return iq
    denominator = 65536
    numerator = round(factor * denominator)
    if not math.isclose(numerator / denominator, factor, rel_tol=1e-9):
        raise SystemExit(
            f"cannot express {source_rate_hz} -> {target_rate_hz} as a small "
            "rational ratio; use a target rate that is an integer multiple"
        )
    common = math.gcd(numerator, denominator)
    return resample_poly(iq, numerator // common, denominator // common)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="source TDMS file")
    parser.add_argument("output", type=Path, help="destination MAT file")
    parser.add_argument(
        "--length-samples",
        type=int,
        default=DEFAULT_LENGTH_SAMPLES,
        help=f"samples to keep at the target rate (default {DEFAULT_LENGTH_SAMPLES})",
    )
    parser.add_argument(
        "--offset-samples",
        type=int,
        default=0,
        help="leading sample offset at the target rate (default 0, slot boundary)",
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=DEFAULT_TARGET_RATE_HZ,
        help=f"closed-loop sample rate in Hz (default {DEFAULT_TARGET_RATE_HZ})",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="print the TDMS metadata and exit",
    )
    args = parser.parse_args(argv)

    import numpy as np
    from scipy.io import savemat

    iq, iq_rate, properties = load_tdms_iq(args.input)
    if args.info:
        print(f"complex samples : {iq.size} at {iq_rate} Hz "
              f"(~{iq.size / iq_rate * 1e3:.3f} ms)")
        for key in (
            "NI_RF_IQRate",
            "NI_RF_SignalBandwidth",
            "NI_RF_WaveformType",
            "NI_RF_PAPR",
            "NI_RF_RuntimeScaling",
        ):
            if key in properties:
                print(f"{key:<24}: {properties[key]}")
        print(f"target rate     : {args.target_rate} Hz")
        return 0

    if args.length_samples <= 0 or args.offset_samples < 0:
        raise SystemExit("--length-samples must be positive and offset non-negative")

    resampled = resample_to_rate(iq, iq_rate, args.target_rate)
    end = args.offset_samples + args.length_samples
    if end > resampled.size:
        raise SystemExit(
            f"requested span [{args.offset_samples}, {end}) exceeds the "
            f"available {resampled.size} resampled samples"
        )

    segment = np.ascontiguousarray(resampled[args.offset_samples : end])
    if not np.all(np.isfinite(segment)):
        raise SystemExit("selected span contains non-finite samples")

    savemat(
        args.output,
        {"x": segment.reshape(-1, 1)},
        do_compression=False,
        oned_as="column",
    )
    peak = float(np.max(np.abs(segment)))
    rms = float(np.sqrt(np.mean(np.abs(segment) ** 2)))
    print(
        f"wrote {args.output}: {segment.size} complex samples "
        f"at {args.target_rate} Hz "
        f"(~{segment.size / args.target_rate * 1e3:.3f} ms), "
        f"peak |x|={peak:.6f}, rms |x|={rms:.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
