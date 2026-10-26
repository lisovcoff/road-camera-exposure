"""Read-only observation helpers. No camera imports, controls or network I/O."""
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil

import numpy as np

from .core import ExposureController, FrameMetadata


class _ShadowPolicy(ExposureController):
    def evaluate(self, scene, metadata, speed, now_ns):
        decision = super().update(scene, metadata, speed, now_ns)
        recommendation = decision.command
        if recommendation is not None:
            # A proposal has NOT been dispatched. Retain observation/mode/dwell
            # state, but discard only that proposal, without inventing an ACK.
            if self._pending != recommendation or self._sent_ns is not None:
                raise RuntimeError("unexpected dispatched state in shadow policy")
            self._pending = self._issued_frame_id = None
            decision = replace(decision, command=None, pending_command_id=None,
                               pending_command=None, sent_ns=None,
                               statuses=tuple(s for s in decision.statuses if s != "AWAITING_DISPATCH"))
        record = decision.to_record()
        record["observation_only"] = True
        record["command_dispatched"] = False
        record["would_request"] = (None if recommendation is None else
            {"exposure_us": recommendation.exposure_us,
             "analogue_gain": recommendation.analogue_gain})
        record["statuses"] = list(record["statuses"]) + ["OBSERVATION_ONLY"]
        return record

    def mark_sent(self, *args, **kwargs):
        raise RuntimeError("shadow policy cannot dispatch commands")


class ShadowAdvisor:
    """Public observation interface; intentionally exposes no dispatch method.

    The internal subclass is coupled to the versioned v0.2 state layout and is
    covered by tests. Camera adapters do not access policy internals.
    """
    def __init__(self, hardware, config):
        self._policy = _ShadowPolicy(hardware, config)

    def observe(self, scene, metadata, speed, now_ns):
        return self._policy.evaluate(scene, metadata, speed, now_ns)


def visible_y_plane(array, stream_config):
    """Use configured dimensions and byte stride, never assumed image width."""
    if not isinstance(stream_config, dict) or stream_config.get("format") != "YUV420":
        raise ValueError("configured YUV420 stream required")
    size, stride = stream_config.get("size"), stream_config.get("stride")
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or any(type(v) is not int or v <= 0 or v % 2 for v in size)
            or type(stride) is not int or stride < size[0]):
        raise ValueError("invalid visible YUV420 size/stride")
    width, height = size
    if (not isinstance(array, np.ndarray) or array.dtype != np.uint8
            or array.ndim != 2 or array.shape != (height * 3 // 2, stride)):
        raise ValueError("YUV array shape disagrees with configured stride/size")
    return array[:height, :width]


def frame_from_metadata(metadata):
    """Picamera2 semantics: first-pixel exposure start = readout timestamp - exposure.

    Raw timestamps remain in the observation log. Never add a guessed offset
    to force agreement with the host clock. The core validates age/ordering.
    SensorTimestamp is a surrogate ID, NOT a hardware frame sequence counter.
    """
    if not isinstance(metadata, dict):
        raise ValueError("metadata mapping required")
    required = ("SensorTimestamp", "ExposureTime", "FrameDuration", "AnalogueGain", "DigitalGain")
    if any(key not in metadata for key in required):
        raise ValueError("required camera metadata missing")
    for key in ("SensorTimestamp", "ExposureTime", "FrameDuration"):
        if type(metadata[key]) is not int or metadata[key] <= 0:
            raise ValueError(f"invalid {key}")
    for key in ("AnalogueGain", "DigitalGain"):
        value = metadata[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid {key}")
    start = metadata["SensorTimestamp"] - metadata["ExposureTime"] * 1000
    if start < 0:
        raise ValueError("exposure start precedes clock origin")
    return FrameMetadata(metadata["SensorTimestamp"], start, metadata["ExposureTime"],
                         metadata["AnalogueGain"], metadata["DigitalGain"], metadata["FrameDuration"])


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


class RunBudget:
    """A new bounded experiment. Reserve checked before every disk write."""
    def __init__(self, output, *, max_bytes, reserve_bytes, free_bytes=None):
        if type(max_bytes) is not int or max_bytes <= 0 or type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("invalid storage budget")
        self.path = Path(output)
        self.max_bytes, self.reserve_bytes, self.used = max_bytes, reserve_bytes, 0
        self._free = free_bytes or (lambda: shutil.disk_usage(self.path.parent).free)
        if not self.path.parent.is_dir():
            raise ValueError("experiment parent directory must already exist")
        self._check(0)
        self.path.mkdir(exist_ok=False)

    def _check(self, count):
        if self.used + count > self.max_bytes:
            raise ValueError("experiment byte budget exhausted")
        if self._free() - count < self.reserve_bytes:
            raise ValueError("storage reserve would be violated")

    def available_bytes(self):
        return self._free()

    def open_log(self):
        self._check(0)
        return (self.path / "observations.jsonl").open("xb")

    def write_new(self, name, content):
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("simple output filename required")
        self._check(len(content))
        with (self.path / name).open("xb") as stream:
            stream.write(content)
        self.used += len(content)

    def write_json(self, name, value):
        content = (json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        self.write_new(name, content)

    def append_record(self, stream, record):
        content = (json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        self._check(len(content))
        stream.write(content)
        stream.flush()
        self.used += len(content)
