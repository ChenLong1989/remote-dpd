"""Explicit per-session state, replacing MATLAB base-workspace variables."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


def waveform_fingerprint(waveform: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(waveform))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


@dataclass(slots=True)
class SessionState:
    reference: np.ndarray | None = None
    current_dpd: np.ndarray | None = None
    iteration: int = 1
    last_feedback_id: str | None = None
    last_input_id: str | None = None
    last_output_id: str | None = None
    last_config_id: str | None = None
    feedback_calibration: complex | None = None
    feedback_binding_verified: bool = False
    external_session_id: str | None = None
    last_metrics: dict[str, object] | None = None

    def reset(self) -> None:
        self.reference = None
        self.current_dpd = None
        self.iteration = 1
        self.last_feedback_id = None
        self.last_input_id = None
        self.last_output_id = None
        self.last_config_id = None
        self.feedback_calibration = None
        self.feedback_binding_verified = False
        self.external_session_id = None
        self.last_metrics = None

    def set_reference(self, reference: np.ndarray) -> bool:
        """Set a new input and report whether it starts a new session."""
        identity = waveform_fingerprint(reference)
        if identity == self.last_input_id and self.reference is not None:
            return False
        self.reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
        self.current_dpd = None
        self.iteration = 1
        self.last_feedback_id = None
        self.last_input_id = identity
        self.last_output_id = None
        self.feedback_calibration = None
        self.feedback_binding_verified = False
        self.external_session_id = None
        self.last_metrics = None
        return True
