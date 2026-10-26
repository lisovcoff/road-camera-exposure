"""Normalized integration contract for a new camera; no SDK dependencies."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class Capabilities:
    min_exposure_us: int
    max_exposure_us: int
    min_gain: float
    max_gain: float
    native_gain_with_manual_shutter: bool
    exposure_readback: bool


@dataclass(frozen=True)
class CapturedFrame:
    y: np.ndarray
    exposure_start_ns: int
    exposure_us: int
    analogue_gain: float
    digital_gain: float
    frame_duration_us: int

    @property
    def midpoint_ns(self):
        return self.exposure_start_ns+self.exposure_us*500


class CameraAdapter(Protocol):
    """Methods must document ownership, actual controls and clock mapping.

    y is a copied visible uint8 plane, excluding stride padding. Timestamps
    share time.monotonic_ns()'s clock; exposure is ALWAYS microseconds here.
    Implementations may expose a separate colour/encoded stream.
    """
    def capabilities(self) -> Capabilities: ...
    def read(self) -> CapturedFrame: ...
    def set_shutter(self, exposure_us: int | None) -> None:
        """None restores native shutter; retain native gain when supported."""
        ...
    def close(self) -> None: ...


def exposure_to_device_units(exposure_us,unit_us):
    """Floor to documented device units. Quantization must be read back."""
    if type(exposure_us) is not int or type(unit_us) is not int or exposure_us<=0 or unit_us<=0:
        raise ValueError('positive integer microseconds/unit size required')
    value=exposure_us//unit_us
    if value==0:raise ValueError('exposure below one device unit')
    return value
