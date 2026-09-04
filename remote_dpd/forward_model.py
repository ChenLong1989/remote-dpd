"""FIR-LUT-FIR (FLF) residual forward model for the forward-model ILC runtime.

The structure is ported from the dpd-compass repository (``models/flf.py``,
documented in its ``docs/algorithm_design.md`` section 13): a fixed 2x
polyphase FIR front end, the reference S-matrix memory tap set with its
nested selection ladder, a piecewise-linear amplitude LUT on a uniform grid,
and per-phase complex coefficients.  The model is linear in its
coefficients, so it is written as ``z_hat = y + Phi(y) w`` and fitted by a
plain blockwise Tikhonov least-squares solve plus a first-difference
penalty on neighbouring LUT knots (see the ridge discussion in
``docs/algorithm_runtime_design.md`` section 2.2).

Two deliberate deviations from the dpd-compass reference, both recorded in
the design document:

- Sequence edges use periodic ``numpy.roll`` semantics instead of the
  zero-extension used there.  ILC waveforms are whole periods by contract,
  so periodic boundaries are exact for this consumer.
- The coefficients are solved directly by least squares; dpd-compass
  trains the same structure with iterative gradient descent.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from remote_dpd.runtime import RuntimeInputError

# Reference S-matrix taps, copied verbatim from dpd-compass models/flf.py
# (origin: getSMatrixMars(716.5)).  Rows are MATLAB reference rows; the
# tables are flattened column-major and the two NaN slots are dropped.
_TAP_X_MATRIX = (
    (0.0, -0.5, -0.5),
    (0.0, -0.5, -0.5),
    (0.0, -0.5, -0.5),
    (0.0, -0.5, math.nan),
    (0.0, 0.0, 0.0),
    (0.0, -0.5, -0.5),
    (0.0, -0.5, -0.5),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (-1.0, -1.0, -1.5),
    (-2.0, -2.0, -2.5),
    (-3.0, -3.0, -3.5),
    (-4.0, -4.0, math.nan),
    (1.0, 1.0, 0.5),
    (2.0, 2.0, 1.5),
    (3.0, 3.0, 2.5),
)
_TAP_P_MATRIX = (
    (0.0, 0.0, -0.5),
    (1.0, 1.0, 0.5),
    (2.0, 2.0, 1.5),
    (3.0, 2.5, math.nan),
    (4.0, 5.0, 6.0),
    (-1.0, -1.0, -1.5),
    (-2.0, -2.0, -2.5),
    (-3.0, -4.0, -5.0),
    (-6.0, -7.0, -8.0),
    (-1.0, -1.5, -1.5),
    (-2.0, -2.5, -2.5),
    (-3.0, -3.5, -3.5),
    (-4.0, -4.5, math.nan),
    (1.0, 0.5, 0.5),
    (2.0, 1.5, 1.5),
    (3.0, 2.5, 2.5),
)


def _flatten_reference_taps() -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    for col in range(3):
        for row in range(16):
            dx = _TAP_X_MATRIX[row][col]
            dp = _TAP_P_MATRIX[row][col]
            if not (math.isnan(dx) or math.isnan(dp)):
                pairs.append((dx, dp))
    return tuple(pairs)


REFERENCE_TAPS = _flatten_reference_taps()

# Nested tap ladder: selection prefix sorted by tap radius then original
# column-major ordinal.  The selected columns stay in original MATLAB order.
_SELECTION_ORDER = tuple(
    sorted(
        range(len(REFERENCE_TAPS)),
        key=lambda i: (max(abs(REFERENCE_TAPS[i][0]), abs(REFERENCE_TAPS[i][1])), i),
    )
)
TAP_COUNTS = (1, 3, 8, 17, 46)

# Fixed polyphase FIRs (center-aligned, 9 taps, centre index 4).  Phase 0 is
# the identity on both input and output; phase 1 is the reference lowpass.
_H_IN_1 = (-0.0239, 0.0670, -0.1705, 0.6217, 0.6217, -0.1705, 0.0670, -0.0239, 0.0)
_H_OUT_1 = (0.0, -0.0239, 0.0670, -0.1705, 0.6217, 0.6217, -0.1705, 0.0670, -0.0239)
_NZ_IN_1 = tuple(range(8))
_NZ_OUT_1 = tuple(range(1, 9))

# G0/H0 are identities, so the receptive field of the output FIR is the
# phase-1 shift range 4-8..4-1 = -4..+3; block assembly slices this margin.
_FIR_MARGIN = 4

# Upper bound on basis samples held at once while fitting (same budget as
# the previous memory-polynomial implementation).
_BLOCK_TERMS = 8_000_000


def tap_pairs_for_count(tap_count: int) -> tuple[tuple[float, float], ...]:
    """Return the selected taps in original MATLAB column-major order."""
    if tap_count not in TAP_COUNTS:
        raise ValueError(f"tap_count must be one of {TAP_COUNTS}, got {tap_count}")
    selected = set(_SELECTION_ORDER[:tap_count])
    return tuple(REFERENCE_TAPS[i] for i in range(len(REFERENCE_TAPS)) if i in selected)


def _roll(signal: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(signal, shift)


def _fir_center(
    signal: np.ndarray,
    taps: tuple[float, ...],
    nonzero: tuple[int, ...],
    axis: int = 0,
) -> np.ndarray:
    """Center-aligned periodic FIR: out[n] = sum_j taps[j] * a[n + (4 - j)]."""
    output = np.zeros_like(signal)
    for j in nonzero:
        if taps[j] != 0.0:
            output = output + taps[j] * np.roll(signal, -(4 - j), axis=axis)
    return output


@dataclass(frozen=True, slots=True)
class FLFModelFit:
    """Least-squares fit of the residual model on one (y, z) pair."""

    coefficients: np.ndarray
    residual_rms: float


class FLFResidualModel:
    """Feature basis, least-squares solver, and adjoint of the FLF model.

    Column layout of ``w``: ``alpha(2, U)`` linear-family coefficients
    (phase-major), then ``beta(2, T, Q-2)`` LUT coefficients (phase, tap in
    original column-major order, interior knot).  With ``U = |unique(d_x)|``
    and ``T = tap_count`` the total complex column count is
    ``2 * (U + T * (Q - 2))``.
    """

    def __init__(self, tap_count: int = 17, lut_size: int = 32) -> None:
        if isinstance(tap_count, bool) or not isinstance(tap_count, numbers.Integral):
            raise ValueError(f"tap_count must be one of {TAP_COUNTS}, got {tap_count!r}")
        tap_count = int(tap_count)
        if tap_count not in TAP_COUNTS:
            raise ValueError(f"tap_count must be one of {TAP_COUNTS}, got {tap_count}")
        if not isinstance(lut_size, int) or isinstance(lut_size, bool) or lut_size < 3:
            raise ValueError(f"lut_size must be an integer >= 3, got {lut_size!r}")
        self.tap_count = tap_count
        self.lut_size = lut_size
        taps = tap_pairs_for_count(tap_count)
        self.tap_x = tuple(dx for dx, _ in taps)
        self.tap_p = tuple(dp for _, dp in taps)
        self.linear_taps = tuple(sorted(set(self.tap_x)))
        self.linear_count = len(self.linear_taps)
        self.tap_total = len(taps)
        values = set(self.linear_taps) | set(self.tap_p)
        self.internal_offsets = tuple(
            sorted({bound for value in values for bound in (math.floor(value), math.ceil(value))})
        )

    def n_columns(self) -> int:
        return 2 * (self.linear_count + self.tap_total * (self.lut_size - 2))

    # --- feature construction -------------------------------------------------

    def phase_bases(self, y: np.ndarray) -> tuple[dict[int, dict[int, np.ndarray]], np.ndarray]:
        """Decimated polyphase bases: bases[r][o][n] = u[2n + r + o].

        u[2m] = y[m]; u[2m+1] = H1{y}[m].  For v = r + o the parity of v
        picks the source array and the shift is v // 2 (floor division keeps
        negative odd offsets on the u1 grid).
        """
        u1 = _fir_center(y, _H_IN_1, _NZ_IN_1)
        bases: dict[int, dict[int, np.ndarray]] = {0: {}, 1: {}}
        for r in (0, 1):
            for offset in self.internal_offsets:
                v = r + offset
                source = y if v % 2 == 0 else u1
                bases[r][offset] = _roll(source, -(v // 2))
        return bases, u1

    def _x_tap(self, bases: dict[int, dict[int, np.ndarray]], r: int, tap: float) -> np.ndarray:
        lo, hi = math.floor(tap), math.ceil(tap)
        if lo == hi:
            return bases[r][lo]
        return 0.5 * (bases[r][lo] + bases[r][hi])

    def _a_tap(self, bases: dict[int, dict[int, np.ndarray]], r: int, tap: float) -> np.ndarray:
        # Magnitudes are averaged, not the magnitude of an averaged sample.
        lo, hi = math.floor(tap), math.ceil(tap)
        if lo == hi:
            return np.abs(bases[r][lo])
        return 0.5 * (np.abs(bases[r][lo]) + np.abs(bases[r][hi]))

    def _out_fir(self, r: int, features: np.ndarray, axis: int = 0) -> np.ndarray:
        """Apply the output FIR of phase r on the decimated grid."""
        if r == 0:
            return features
        return _fir_center(features, _H_OUT_1, _NZ_OUT_1, axis=axis)

    def lut_weights(self, amplitude: np.ndarray) -> tuple[np.ndarray, ...]:
        """Sparse two-knot weights (lo, hi, w_lo, w_hi) on the interior knots.

        Endpoint knots 0 and Q-1 are fixed zero; amplitudes above (Q-1)/Q
        deactivate all hats (dpd-compass section 13.8 semantics).
        """
        q = self.lut_size
        scaled = amplitude * q
        lo = np.clip(np.floor(scaled).astype(np.int64), 0, q - 2)
        frac = scaled - lo
        valid = (amplitude >= 0.0) & (amplitude <= (q - 1.0) / q)
        w_lo = (1.0 - frac) * valid * (lo >= 1)
        w_hi = frac * valid * (lo + 1 <= q - 2)
        return lo, lo + 1, w_lo, w_hi

    def _dense_hats(self, amplitude: np.ndarray) -> np.ndarray:
        """All interior hat values as (samples, Q-2), analytic definition."""
        q = self.lut_size
        knots = np.arange(1, q - 1, dtype=amplitude.dtype) / q
        return np.clip(1.0 - np.abs(amplitude[:, None] - knots[None, :]) * q, 0.0, None)

    def _block_matrix(
        self,
        bases: dict[int, dict[int, np.ndarray]],
        x_linear: dict[int, list[np.ndarray]],
        x_taps: dict[int, list[np.ndarray]],
        a_taps: dict[int, list[np.ndarray]],
        start: int,
        stop: int,
    ) -> np.ndarray:
        """Assemble the (stop-start, K) feature block for the sample range.

        The phase-1 output FIR needs +-4 samples of margin; the margin is
        sliced away after filtering so blocks stitch exactly.
        """
        count = self.n_columns()
        length = stop - start
        total = bases[0][0].size
        block = np.zeros((length, count), dtype=np.complex128)
        column = 0
        for r in (0, 1):
            for i in range(self.linear_count):
                block[:, column] = x_linear[r][i][start:stop]
                column += 1
        margin = _FIR_MARGIN
        for r in (0, 1):
            for t in range(self.tap_total):
                if r == 0:
                    x = x_taps[0][t][start:stop]
                    features = x[:, None] * self._dense_hats(a_taps[0][t][start:stop])
                else:
                    index = np.arange(start - margin, stop + margin) % total
                    x_window = x_taps[1][t][index]
                    window = x_window[:, None] * self._dense_hats(a_taps[1][t][index])
                    features = _fir_center(window, _H_OUT_1, _NZ_OUT_1, axis=0)[margin : margin + length]
                block[:, column : column + self.lut_size - 2] = features
                column += self.lut_size - 2
        return block

    def _tap_arrays(
        self, y: np.ndarray
    ) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, list[np.ndarray]], dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
        bases, _ = self.phase_bases(y)
        x_linear = {
            r: [self._x_tap(bases, r, d) for d in self.linear_taps] for r in (0, 1)
        }
        x_taps = {r: [self._x_tap(bases, r, d) for d in self.tap_x] for r in (0, 1)}
        a_taps = {r: [self._a_tap(bases, r, d) for d in self.tap_p] for r in (0, 1)}
        x_linear = {
            r: [_fir_center(x_linear[r][i], _H_OUT_1, _NZ_OUT_1) if r else x_linear[r][i] for i in range(self.linear_count)]
            for r in (0, 1)
        }
        return bases, x_linear, x_taps, a_taps

    # --- least squares ---------------------------------------------------------

    def fit(
        self,
        y: np.ndarray,
        z: np.ndarray,
        ridge: float,
        lut_ridge: float,
    ) -> FLFModelFit:
        """Fit z ~ y + Phi(y) w by regularized complex least squares.

        Minimizes ||z - y - Phi w||^2 + ridge ||w||^2 + lut_ridge ||D beta||^2
        where D is the first difference along each beta knot axis.  The plain
        ridge guards the (structurally rank-deficient) basis; the difference
        penalty suppresses adjacent-knot jumps whose hat-slope amplification
        spikes the adjoint at waveform peaks.  Gram and right-hand side are
        accumulated blockwise; non-finite energies fail closed.
        """
        count = self.n_columns()
        total = y.size
        target = z - y
        bases, x_linear, x_taps, a_taps = self._tap_arrays(y)
        gram = np.zeros((count, count), dtype=np.complex128)
        rhs = np.zeros(count, dtype=np.complex128)
        target_energy = float(np.vdot(target, target).real)
        block = max(1, _BLOCK_TERMS // count)
        for start in range(0, total, block):
            stop = min(start + block, total)
            matrix = self._block_matrix(bases, x_linear, x_taps, a_taps, start, stop)
            if not np.all(np.isfinite(matrix)):
                raise RuntimeInputError(
                    "forward model basis produced non-finite samples for this input"
                )
            gram += matrix.conj().T @ matrix
            rhs += matrix.conj().T @ target[start:stop]
        if not math.isfinite(target_energy) or not np.all(np.isfinite(gram)):
            raise RuntimeInputError(
                "forward model basis overflowed for this input scale"
            )
        gram[np.diag_indices_from(gram)] += ridge
        self._add_lut_difference_penalty(gram, lut_ridge)
        try:
            coefficients = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]

        model_energy = float(np.vdot(coefficients, gram @ coefficients).real)
        cross_energy = 2.0 * float(np.vdot(coefficients, rhs).real)
        residual_energy = max(0.0, target_energy - cross_energy + model_energy)
        return FLFModelFit(
            coefficients=coefficients,
            residual_rms=math.sqrt(residual_energy / total),
        )

    def _add_lut_difference_penalty(self, gram: np.ndarray, lut_ridge: float) -> None:
        """Add lut_ridge * D^H D on each beta knot axis in place."""
        if lut_ridge <= 0.0:
            return
        q = self.lut_size
        knots = q - 2
        for r in (0, 1):
            for t in range(self.tap_total):
                base = 2 * self.linear_count + r * self.tap_total * knots + t * knots
                for k in range(knots):
                    diagonal = 2.0 if 0 < k < knots - 1 else 1.0
                    gram[base + k, base + k] += lut_ridge * diagonal
                    if k + 1 < knots:
                        gram[base + k, base + k + 1] -= lut_ridge
                        gram[base + k + 1, base + k] -= lut_ridge

    # --- forward evaluation (tests and replay verification) -------------------

    def evaluate(self, y: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        """Evaluate z_hat = y + Phi(y) w at full length."""
        bases, x_linear, x_taps, a_taps = self._tap_arrays(y)
        total = y.size
        output = np.zeros_like(y)
        block = max(1, _BLOCK_TERMS // self.n_columns())
        for start in range(0, total, block):
            stop = min(start + block, total)
            matrix = self._block_matrix(bases, x_linear, x_taps, a_taps, start, stop)
            output[start:stop] = matrix @ coefficients
        return y + output

    # --- adjoint gradient -------------------------------------------------------

    def adjoint(
        self,
        y: np.ndarray,
        error: np.ndarray,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """Back-propagate the output error through the fitted model Jacobian.

        The residual model F(y) = y + Phi(y) w has A = I + A_Phi and
        B = B_Phi, so the real-parameterization steepest-descent direction of
        ||F(y) - x||^2 is g = e + A_Phi^H e + conj(B_Phi^H e).  The identity
        contributes the direct term e; the FLF path propagates through the
        output FIR adjoint, then each tap splits into a holomorphic signal
        path (weight conj(LUT(A))) and an amplitude path (weight
        conj(X) conj(dLUT/da) times the Wirtinger derivatives of |u|), and
        finally through the input-FIR adjoint back to y.  Half-integer taps
        scatter 0.5 onto each of their two integer endpoints.
        """
        bases, _ = self.phase_bases(y)
        u = self.linear_count
        t_total = self.tap_total
        q = self.lut_size
        alpha = coefficients[: 2 * u].reshape(2, u)
        beta = coefficients[2 * u :].reshape(2, t_total, q - 2)
        beta_ext = np.concatenate(
            (np.zeros((2, t_total, 1)), beta, np.zeros((2, t_total, 1))), axis=2
        )

        g_a_y = error.copy()  # identity path (G0/H0 identity phases)
        g_a_u1 = np.zeros_like(y)
        g_b_y = np.zeros_like(y)
        g_b_u1 = np.zeros_like(y)

        # Adjoint of the output FIR on the decimated phase grids (real taps,
        # so the adjoint is the flipped convolution).
        v1 = np.zeros_like(error)
        for j in _NZ_OUT_1:
            v1 = v1 + _H_OUT_1[j] * _roll(error, (4 - j))
        v_phase = [error, v1]

        def scatter(r: int, offset: int, weight: np.ndarray, part_a: bool) -> None:
            nonlocal g_a_y, g_a_u1, g_b_y, g_b_u1
            v = r + offset
            if part_a:
                if v % 2 == 0:
                    g_a_y += _roll(weight, (v // 2))
                else:
                    g_a_u1 += _roll(weight, (v // 2))
            else:
                if v % 2 == 0:
                    g_b_y += _roll(weight, (v // 2))
                else:
                    g_b_u1 += _roll(weight, (v // 2))

        def scatter_tap(r: int, tap: float, weight: np.ndarray) -> None:
            lo, hi = math.floor(tap), math.ceil(tap)
            if lo == hi:
                scatter(r, lo, weight, True)
            else:
                scatter(r, lo, 0.5 * weight, True)
                scatter(r, hi, 0.5 * weight, True)

        for r in (0, 1):
            v_r = v_phase[r]
            for i, delay in enumerate(self.linear_taps):
                scatter_tap(r, delay, v_r * np.conj(alpha[r, i]))
            for t in range(t_total):
                dx, dp = self.tap_x[t], self.tap_p[t]
                x = self._x_tap(bases, r, dx)
                a = self._a_tap(bases, r, dp)
                lo, hi, w_lo, w_hi = self.lut_weights(a)
                lut = beta_ext[r, t, lo] * w_lo + beta_ext[r, t, hi] * w_hi
                valid = (a >= 0.0) & (a <= (q - 1.0) / q)
                slope = (q * (beta_ext[r, t, hi] - beta_ext[r, t, lo])) * valid
                # Signal path: holomorphic in the X sample.
                scatter_tap(r, dx, v_r * np.conj(lut))
                # Amplitude path: A_dp = share * (|b_lo| [+ |b_hi|]); each
                # magnitude contributes 0.5 u/|u| to dA/du and 0.5 conj(u)/|u|
                # to dA/du* (halved again for half-integer taps).
                endpoints = (
                    ((math.floor(dp), 1.0),)
                    if math.floor(dp) == math.ceil(dp)
                    else ((math.floor(dp), 0.5), (math.ceil(dp), 0.5))
                )
                for endpoint, share in endpoints:
                    sample = bases[r][endpoint]
                    magnitude = np.abs(sample)
                    unit = np.zeros_like(sample)
                    nonzero = magnitude > 0
                    unit[nonzero] = sample[nonzero] / magnitude[nonzero]
                    common = v_r * np.conj(x) * np.conj(slope) * share * 0.5
                    scatter(r, endpoint, common * unit, True)
                    scatter(r, endpoint, common * np.conj(unit), False)

        # Backprop through the phase-1 input FIR for both parts (real taps).
        for j in _NZ_IN_1:
            g_a_y = g_a_y + _H_IN_1[j] * _roll(g_a_u1, (4 - j))
            g_b_y = g_b_y + _H_IN_1[j] * _roll(g_b_u1, (4 - j))
        return g_a_y + np.conj(g_b_y)
