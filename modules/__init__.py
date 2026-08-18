"""Core modules for the Auto Sort Pi vision and robot-control pipeline."""

from .camera_stream import CameraStream
from .detector import WasteDetector
from .logging_setup import configure_logging, get_logger, preview
from .scheduler import optimize_trajectory
from .serial_comm import SerialCommunicator
from .transformer import CoordinateTransformer

__all__ = [
    "CameraStream",
    "WasteDetector",
    "optimize_trajectory",
    "SerialCommunicator",
    "CoordinateTransformer",
    "configure_logging",
    "get_logger",
    "preview",
]
