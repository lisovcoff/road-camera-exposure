"""Picamera2 laboratory setup and metadata helpers; no camera import at module load."""

import hashlib
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..imaging import json_safe

def configure_lab(camera):
    """Same optical/sensor pipeline as observer; explicit manual AE/AF."""
    if "imx708" not in str(camera.camera_properties.get("Model", "")).lower():
        raise ValueError("identified IMX708 required")
    requested = camera.create_video_configuration(
        main={"size": (1920, 1080), "format": "YUV420"},
        lores={"size": (320, 180), "format": "YUV420"}, raw=None,
        display=None, queue=False,
        sensor={"output_size": (2304, 1296), "bit_depth": 10},
        controls={"FrameRate": 30, "AeEnable": False, "ExposureTime": 2000,
                  "AnalogueGain": 2.0, "AfMode": 0, "LensPosition": 1.0})
    requested_copy = json_safe(requested)
    camera.configure(requested)
    actual = camera.camera_configuration()
    for stream, size in (("main", (1920, 1080)), ("lores", (320, 180))):
        if tuple(actual[stream]["size"]) != size or actual[stream]["format"] != "YUV420":
            raise ValueError("unexpected stream configuration")
    if (tuple(actual["sensor"]["output_size"]) != (2304, 1296)
            or actual["sensor"]["bit_depth"] != 10
            or "identity" not in str(actual.get("transform"))):
        raise ValueError("unexpected sensor/transform")
    exp, gain = camera.camera_controls["ExposureTime"], camera.camera_controls["AnalogueGain"]
    if not (exp[0] <= 100 <= 12000 <= exp[1] and gain[0] <= 2 <= 16 <= gain[1]):
        raise ValueError("required bounded controls unavailable")
    return requested_copy, actual

def setup_record(camera, requested, actual):
    return {"requested_camera_configuration": requested, "actual_camera_configuration": actual,
            "camera_properties": camera.camera_properties, "camera_controls": camera.camera_controls,
            "versions": package_versions(), "system_before": system_metrics(),
            "source_sha256": {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path(__file__).parents[1].rglob('*.py'))},
            "road_calibrated": False, "sensor_hdr_state": "not explicitly changed or independently verified",
            "focus": "manual; see session controls, NOT calibrated", "speed_source": "session-specific"}

def system_metrics():
    result = {"checked_monotonic_ns": time.monotonic_ns()}
    for label, command in (("throttled", ["vcgencmd", "get_throttled"]),
                           ("temperature", ["vcgencmd", "measure_temp"])):
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=2)
            result[label] = completed.stdout.strip() if completed.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            result[label] = "unavailable"
    return result

def package_versions():
    try:
        result = subprocess.run(["dpkg-query", "-W", "python3-picamera2", "python3-libcamera",
                                 "libcamera0.7", "rpicam-apps-core", "rpicam-apps-lite"],
                                capture_output=True, text=True, timeout=2)
        return {"python": platform.python_version(), "numpy": np.__version__,
                "dpkg_query_returncode": result.returncode, "packages": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired):
        return {"python": platform.python_version(), "numpy": np.__version__, "packages": "unavailable"}

def validate_roi(roi):
    if not isinstance(roi, (tuple, list)) or len(roi) != 4 or any(type(x) is not int for x in roi):
        raise ValueError('ROI must be integer x0 y0 x1 y1')
    x0,y0,x1,y1=roi
    if not (0<=x0<x1<=1920 and 0<=y0<y1<=1080 and 8<=x1-x0<=64 and 8<=y1-y0<=64):
        raise ValueError('main-image patch must be 8..64 pixels per side')

@dataclass(frozen=True)
class TrialSpec:
    mode: str
    target_roi: tuple
    background_roi: tuple
    colour_gains: tuple
    lens: float
    exposure_us: int = 4000
    duration_s: int = 10

    def __post_init__(self):
        if self.mode not in ('native-normal','native-short','shutter-auto-gain'):
            raise ValueError('unsupported trial mode')
        for roi in (self.target_roi,self.background_roi):
            validate_roi(roi)
        a,b=self.target_roi,self.background_roi
        if max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3]):
            raise ValueError('target/background patches must not overlap')
        if (not isinstance(self.colour_gains,(tuple,list)) or len(self.colour_gains)!=2
                or any(type(v) not in (int,float) or not math.isfinite(v) or not .1<=v<=8 for v in self.colour_gains)):
            raise ValueError('two measured colour gains required in .1..8')
        if type(self.lens) not in (float,int) or not math.isfinite(self.lens) or not 0<=self.lens<=2:
            raise ValueError('lens candidate must be in 0..2')
        if type(self.exposure_us) is not int or not 100<=self.exposure_us<=12000:
            raise ValueError('fixed shutter experiment limited to 100..12000 us')
        if type(self.duration_s) is not int or not 5<=self.duration_s<=30:
            raise ValueError('duration must be 5..30 seconds')

    def controls(self):
        manual=self.mode=='shutter-auto-gain'
        return {'AeEnable':True, 'ExposureTime':self.exposure_us if manual else 0,
                'AnalogueGain':0, 'ExposureTimeMode':1 if manual else 0, 'AnalogueGainMode':0,
                'AeExposureMode':1 if self.mode=='native-short' else 0,
                'AeMeteringMode':0, 'AeConstraintMode':0, 'ExposureValue':0.,
                'AfMode':0, 'LensPosition':self.lens,
                'AwbEnable':False, 'ColourGains':tuple(self.colour_gains)}

def check_readback(md,spec):
    """Mode fields may be emitted only when changed; absence is explicit."""
    reported = md.get('ExposureTimeMode'),md.get('AnalogueGainMode')
    wanted = (1 if spec.mode=='shutter-auto-gain' else 0,0)
    if any(a is not None and a!=b for a,b in zip(reported,wanted)):
        return False,'wrong_control_mode'
    if md.get('AeState') not in (1,2):
        return False,'native_AE_not_active_or_unreported'
    if (spec.mode=='shutter-auto-gain' and abs(md['ExposureTime']-spec.exposure_us)>42):
        return False,'fixed_shutter_not_observed'
    if md.get('LensPosition') is None or abs(md['LensPosition']-spec.lens)>.025:
        return False,'focus_not_observed'
    gains=md.get('ColourGains')
    if gains is None or len(gains)!=2 or any(abs(a-b)>.05 for a,b in zip(gains,spec.colour_gains)):
        return False,'fixed_AWB_not_observed'
    return True,'reported_modes' if None not in reported else 'AE_active_modes_partly_unreported'

METADATA_FIELDS = (
    "SensorTimestamp", "ExposureTime", "AnalogueGain", "DigitalGain", "FrameDuration",
    "LensPosition", "AfState", "FocusFoM", "AeState", "Lux", "ScalerCrop",
    "ColourGains", "ColourTemperature", "HdrChannel", "ExposureTimeMode", "AnalogueGainMode",
)
