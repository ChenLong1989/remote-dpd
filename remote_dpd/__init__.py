"""Device-driven DPD simulation, orchestration, and file command service."""

from .controller import (
    DEFAULT_NORMALIZE_REFERENCE_RMS,
    DEFAULT_REFERENCE_TARGET_RMS_DBFS,
    ClosedLoopConfig,
    ClosedLoopController,
    ControllerBusyError,
    ControllerError,
    ControllerSnapshot,
    ControllerState,
    ControllerStateError,
    ControllerStoppedError,
    IterationRecord,
    ReferenceNormalizationReport,
)
from .device import (
    CaptureRequest,
    DeviceConfig,
    DeviceRegistrationError,
    RFBench,
    create_rf_bench,
    list_rf_benches,
    register_rf_bench,
)
from .file_interface import (
    CommandStatus,
    FileCommandError,
    FileCommandProcessor,
    FileCommandService,
    parse_configuration_json,
)
from .power_control import (
    PowerAdjustment,
    PowerControlCancelled,
    PowerControlError,
    PowerController,
    PowerControlResult,
)
from .preprocessing import CaptureBatch, FeedbackPreprocessor, PreprocessingResult
from .result_export import (
    ResultExportError,
    build_final_payload,
    export_final_mat,
)
from .runtime import (
    BasicILCRuntime,
    DPDRuntime,
    RuntimeStepInput,
    RuntimeStepResult,
    create_runtime,
)
from .safety import (
    DigitalSafetyError,
    DigitalSafetyReport,
    validate_candidate,
    validate_reference,
)
from .simulation import SIMULATED_DEVICE_SCHEMA, SimulatedRFBench
from .storage import RunRecorder, RunStorageError, RunStore
from .waveforms import WaveformAccessError, WaveformRepository
from .web import create_web_app
from .web_bridge import WebBridgeError, WebCommandBridge

__all__ = [
    "DEFAULT_NORMALIZE_REFERENCE_RMS",
    "DEFAULT_REFERENCE_TARGET_RMS_DBFS",
    "SIMULATED_DEVICE_SCHEMA",
    "BasicILCRuntime",
    "CaptureBatch",
    "CaptureRequest",
    "ClosedLoopConfig",
    "ClosedLoopController",
    "CommandStatus",
    "ControllerBusyError",
    "ControllerError",
    "ControllerSnapshot",
    "ControllerState",
    "ControllerStateError",
    "ControllerStoppedError",
    "DPDRuntime",
    "DeviceConfig",
    "DeviceRegistrationError",
    "DigitalSafetyError",
    "DigitalSafetyReport",
    "FeedbackPreprocessor",
    "FileCommandError",
    "FileCommandProcessor",
    "FileCommandService",
    "IterationRecord",
    "PowerAdjustment",
    "PowerControlCancelled",
    "PowerControlError",
    "PowerControlResult",
    "PowerController",
    "PreprocessingResult",
    "RFBench",
    "ReferenceNormalizationReport",
    "ResultExportError",
    "RunRecorder",
    "RunStorageError",
    "RunStore",
    "RuntimeStepInput",
    "RuntimeStepResult",
    "SimulatedRFBench",
    "WaveformAccessError",
    "WaveformRepository",
    "WebBridgeError",
    "WebCommandBridge",
    "build_final_payload",
    "create_rf_bench",
    "create_runtime",
    "create_web_app",
    "export_final_mat",
    "list_rf_benches",
    "parse_configuration_json",
    "register_rf_bench",
    "validate_candidate",
    "validate_reference",
]
