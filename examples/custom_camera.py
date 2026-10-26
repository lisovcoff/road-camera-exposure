"""SDK integration template, not a working USB/IP camera driver.

Implement read/set_shutter/capabilities using your documented vendor API.
Start with read-only logging and actual readback before enabling control.
"""
from road_camera_exposure.adapters.base import Capabilities,CapturedFrame


class VendorCamera:
    def capabilities(self) -> Capabilities:
        raise NotImplementedError('Query actual exposure/gain bounds and native-gain support')

    def read(self) -> CapturedFrame:
        raise NotImplementedError('Copy visible Y; normalize actual controls and sensor clock')

    def set_shutter(self,exposure_us: int | None) -> None:
        raise NotImplementedError('Convert microseconds to documented units; confirm on fresh frames')

    def close(self) -> None:
        raise NotImplementedError('Release requests, encoder and device even after failure')
