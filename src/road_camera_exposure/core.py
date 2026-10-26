"""Camera-independent experimental exposure policy. No I/O or camera imports.

All thresholds are supplied by the caller, not calibrated defaults. Times are
integer nanoseconds on ONE monotonic clock. Frame timestamps mean first-pixel
exposure START; a future adapter must convert sensor metadata appropriately.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


def _number(value, name, low, high=math.inf):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name}: expected finite number in [{low}, {high}]")


def _integer(value, name, low=0):
    if type(value) is not int or not low <= value <= 2**63 - 1:
        raise ValueError(f"{name}: expected integer >= {low}")


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected nonempty string")


@dataclass(frozen=True)
class ZoneMask:
    name: str
    mask: np.ndarray
    weight: float

    def __post_init__(self):
        _label(self.name, "zone name")
        _number(self.weight, "zone weight", np.finfo(float).tiny)
        if not isinstance(self.mask, np.ndarray) or self.mask.ndim != 2 or self.mask.dtype != np.bool_:
            raise ValueError("zone mask must be a 2D boolean array")
        mask = self.mask.copy()
        mask.flags.writeable = False
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class AnalysisConfig:
    dark_y: int
    clipped_y: int
    metering_percentile: float
    min_zone_pixels: int

    def __post_init__(self):
        _integer(self.dark_y, "dark_y")
        _integer(self.clipped_y, "clipped_y")
        if not self.dark_y < self.clipped_y <= 255:
            raise ValueError("require dark_y < clipped_y <= 255")
        _number(self.metering_percentile, "metering percentile", 0, 100)
        _integer(self.min_zone_pixels, "min_zone_pixels", 1)


@dataclass(frozen=True)
class ZoneStats:
    name: str
    pixels: int
    weight: float
    meter_y: float
    median: float
    p90: float
    p99: float
    dark_pct: float
    clipped_pct: float


@dataclass(frozen=True)
class SceneStats:
    mask_version: str
    transform: str
    calibrated: bool
    meter_y: float
    dark_pct: float
    clipped_pct: float
    zones: tuple[ZoneStats, ...]


def analyze_y_plane(y_plane, zones, config: AnalysisConfig, *, mask_version,
                    transform, calibrated=False, exclude=None) -> SceneStats:
    """Analyze visible Y only. Caller removes YUV padding using configured size.

    Masks are non-overlapping pixel masks in the supplied image coordinates.
    Each zone has a percentile meter; small glare regions do not replace the
    road meter. Clipping is also measured separately, before any aggregation.
    """
    if not isinstance(config, AnalysisConfig):
        raise ValueError("AnalysisConfig required")
    if (not isinstance(y_plane, np.ndarray) or y_plane.ndim != 2
            or y_plane.dtype != np.uint8 or not y_plane.size):
        raise ValueError("expected nonempty 2D uint8 visible Y plane")
    _label(mask_version, "mask_version")
    _label(transform, "transform")
    if type(calibrated) is not bool:
        raise ValueError("calibrated must be boolean")
    if exclude is None:
        exclude = np.zeros(y_plane.shape, dtype=bool)
    if (not isinstance(exclude, np.ndarray) or exclude.shape != y_plane.shape
            or exclude.dtype != np.bool_):
        raise ValueError("exclusion mask must be boolean and match visible Y")
    if not isinstance(zones, (tuple, list)) or not zones:
        raise ValueError("at least one explicit zone required")
    occupied = np.zeros(y_plane.shape, dtype=bool)
    names = set()
    result = []
    for zone in zones:
        if not isinstance(zone, ZoneMask) or zone.mask.shape != y_plane.shape:
            raise ValueError("zone mask shape must match visible Y")
        if zone.name in names:
            raise ValueError("zone names must be unique")
        names.add(zone.name)
        active = zone.mask & ~exclude
        if np.any(occupied & active):
            raise ValueError("active zones must not overlap")
        occupied |= active
        values = y_plane[active]
        if values.size < config.min_zone_pixels:
            raise ValueError(f"zone {zone.name}: too few unexcluded pixels")
        meter, median, p90, p99 = np.percentile(
            values, [config.metering_percentile, 50, 90, 99])
        result.append(ZoneStats(
            zone.name, int(values.size), float(zone.weight), float(meter),
            float(median), float(p90), float(p99),
            float(np.mean(values <= config.dark_y) * 100),
            float(np.mean(values >= config.clipped_y) * 100),
        ))
    # Normalize before summation to avoid overflow for large finite weights.
    largest = max(item.weight for item in result)
    weights = [item.weight / largest for item in result]
    total = math.fsum(weights)
    def weighted(field):
        return math.fsum(getattr(item, field) * (weight / total)
                         for item, weight in zip(result, weights))
    return SceneStats(mask_version, transform, calibrated, weighted("meter_y"),
                      weighted("dark_pct"), weighted("clipped_pct"), tuple(result))


@dataclass(frozen=True)
class ExposureLimits:
    min_exposure_us: int
    max_exposure_us: int
    min_gain: float
    max_gain: float
    exposure_tolerance_us: int
    gain_tolerance: float

    def __post_init__(self):
        _integer(self.min_exposure_us, "min exposure", 1)
        _integer(self.max_exposure_us, "max exposure", self.min_exposure_us)
        _number(self.min_gain, "min gain", 1)
        _number(self.max_gain, "max gain", self.min_gain)
        _integer(self.exposure_tolerance_us, "exposure tolerance")
        _number(self.gain_tolerance, "gain tolerance", 0)
        if self.exposure_tolerance_us >= self.min_exposure_us or self.gain_tolerance >= self.min_gain:
            raise ValueError("readback tolerances must be smaller than minimum controls")
        _number(self.max_exposure_us * self.max_gain, "maximum exposure product", 1)


@dataclass(frozen=True)
class Profile:
    min_exposure_us: int
    max_exposure_us: int
    preferred_exposure_us: int
    target_y: float
    band_fraction: float

    def __post_init__(self):
        _integer(self.min_exposure_us, "profile min exposure", 1)
        _integer(self.max_exposure_us, "profile max exposure", self.min_exposure_us)
        _integer(self.preferred_exposure_us, "preferred exposure", self.min_exposure_us)
        if self.preferred_exposure_us > self.max_exposure_us:
            raise ValueError("preferred exposure exceeds profile maximum")
        _number(self.target_y, "target Y", np.finfo(float).tiny, 255)
        _number(self.band_fraction, "brightness band", 0, 0.99)


@dataclass(frozen=True)
class PolicyConfig:
    version: str
    detect: Profile
    view: Profile
    enter_detect_kmh: float
    enter_view_kmh: float
    mode_dwell_ns: int
    status_dwell_ns: int
    max_speed_age_ns: int
    max_frame_age_ns: int
    command_timeout_ns: int
    min_response: float
    max_response: float
    highlight_pct: float

    def __post_init__(self):
        _label(self.version, "policy version")
        if not isinstance(self.detect, Profile) or not isinstance(self.view, Profile):
            raise ValueError("two explicit profiles required")
        _number(self.enter_detect_kmh, "enter DETECT speed", 0)
        _number(self.enter_view_kmh, "enter VIEW speed", 0)
        if self.enter_detect_kmh >= self.enter_view_kmh:
            raise ValueError("speed thresholds must have a hysteresis gap")
        for field in ("mode_dwell_ns", "status_dwell_ns"):
            _integer(getattr(self, field), field)
        for field in ("max_speed_age_ns", "max_frame_age_ns", "command_timeout_ns"):
            _integer(getattr(self, field), field, 1)
        _number(self.min_response, "minimum response", np.finfo(float).tiny, 1)
        _number(self.max_response, "maximum response", 1)
        _number(self.highlight_pct, "highlight percent", np.finfo(float).tiny, 100)


@dataclass(frozen=True)
class SpeedSample:
    kmh: float
    timestamp_ns: int


@dataclass(frozen=True)
class FrameMetadata:
    frame_id: int
    exposure_start_ns: int
    exposure_us: int
    analogue_gain: float
    digital_gain: float
    frame_duration_us: int


@dataclass(frozen=True)
class ExposureCommand:
    command_id: int
    created_ns: int
    exposure_us: int
    analogue_gain: float


@dataclass(frozen=True)
class Decision:
    now_ns: int | None
    policy_version: str
    mode: str
    statuses: tuple[str, ...]
    reason: str
    limits: ExposureLimits
    command: ExposureCommand | None
    pending_command_id: int | None
    pending_command: ExposureCommand | None
    sent_ns: int | None
    observed_command_id: int | None
    observed_sent_ns: int | None
    actual: FrameMetadata | None
    statistics: SceneStats | None
    speed: SpeedSample | None

    def to_record(self):
        """JSON-compatible telemetry, no file writes or hidden camera activity."""
        return {"schema_version": 2, **asdict(self)}


class ExposureController:
    """One command in flight; explicit dispatch and frame-associated readback.

    A timeout is latched. Recovery requires a new controller AFTER the caller
    verifies actual camera state; there is no blind retry or implicit reset.
    """

    def __init__(self, hardware: ExposureLimits, config: PolicyConfig):
        if not isinstance(hardware, ExposureLimits) or not isinstance(config, PolicyConfig):
            raise ValueError("hardware limits and policy configuration required")
        self.hardware, self.config = hardware, config
        for profile in (config.detect, config.view):
            self._effective(profile)
        if config.detect.max_exposure_us > config.view.max_exposure_us:
            raise ValueError("DETECT shutter ceiling must not exceed VIEW ceiling")
        self.mode = "DETECT"
        self._mode_candidate = None
        self._status_candidate = None
        self._pending = None
        self._sent_ns = None
        self._issued_frame_id = None
        self._next_id = 1
        self._last_now = None
        self._last_frame = None
        self._last_frame_time = None
        self._last_speed_time = None
        self._fault = False

    def _effective(self, profile):
        low = max(profile.min_exposure_us, self.hardware.min_exposure_us)
        high = min(profile.max_exposure_us, self.hardware.max_exposure_us)
        if low > high:
            raise ValueError("profile and hardware exposure ranges do not intersect")
        return ExposureLimits(low, high, self.hardware.min_gain, self.hardware.max_gain,
                              min(self.hardware.exposure_tolerance_us, low - 1),
                              self.hardware.gain_tolerance)

    def mark_sent(self, command_id: int, sent_ns: int):
        """Adapter calls once after dispatch; timestamp is dispatch start time."""
        _integer(command_id, "command ID", 1)
        _integer(sent_ns, "dispatch time")
        if (self._fault or self._pending is None or command_id != self._pending.command_id
                or self._sent_ns is not None or sent_ns < self._pending.created_ns
                or (self._last_now is not None and sent_ns < self._last_now)):
            raise ValueError("command is not awaiting this dispatch")
        self._sent_ns = sent_ns

    def _same_controls(self, metadata, command):
        return (abs(metadata.exposure_us - command.exposure_us) <= self.hardware.exposure_tolerance_us
                and abs(metadata.analogue_gain - command.analogue_gain) <= self.hardware.gain_tolerance)

    def _validate(self, scene, metadata, speed, now):
        if not isinstance(scene, SceneStats) or not isinstance(metadata, FrameMetadata):
            raise ValueError("SceneStats and FrameMetadata required")
        _label(scene.mask_version, "mask version")
        _label(scene.transform, "transform")
        if type(scene.calibrated) is not bool or not scene.zones:
            raise ValueError("scene requires calibration flag and zone statistics")
        for value, name, high in ((scene.meter_y, "meter Y", 255),
                                  (scene.dark_pct, "dark percent", 100),
                                  (scene.clipped_pct, "clipped percent", 100)):
            _number(value, name, 0, high)
        for zone in scene.zones:
            if not isinstance(zone, ZoneStats):
                raise ValueError("invalid zone statistics")
            _label(zone.name, "zone name")
            _integer(zone.pixels, "zone pixels", 1)
            _number(zone.weight, "zone weight", np.finfo(float).tiny)
            for field in ("meter_y", "median", "p90", "p99", "dark_pct", "clipped_pct"):
                _number(getattr(zone, field), field, 0, 100 if field.endswith("pct") else 255)
        _integer(metadata.frame_id, "frame ID")
        _integer(metadata.exposure_start_ns, "frame time")
        _integer(metadata.exposure_us, "actual exposure", 1)
        _integer(metadata.frame_duration_us, "frame duration", 1)
        _number(metadata.analogue_gain, "actual analogue gain", 1)
        _number(metadata.digital_gain, "actual digital gain", np.finfo(float).tiny)
        _number(metadata.exposure_us * metadata.analogue_gain, "actual exposure product", 1)
        if not (self.hardware.min_exposure_us - self.hardware.exposure_tolerance_us
                <= metadata.exposure_us <= self.hardware.max_exposure_us + self.hardware.exposure_tolerance_us
                and self.hardware.min_gain - self.hardware.gain_tolerance
                <= metadata.analogue_gain <= self.hardware.max_gain + self.hardware.gain_tolerance):
            raise ValueError("readback lies outside hardware limits and tolerance")
        if metadata.exposure_us > metadata.frame_duration_us:
            raise ValueError("exposure exceeds frame duration")
        if metadata.exposure_start_ns > now:
            raise ValueError("frame is from the future or another clock")
        if speed is not None:
            if not isinstance(speed, SpeedSample):
                raise ValueError("SpeedSample or None required")
            _number(speed.kmh, "speed", 0)
            _integer(speed.timestamp_ns, "speed timestamp")
            if speed.timestamp_ns > now:
                raise ValueError("speed is from the future or another clock")

    def _update_mode(self, speed, now, stale):
        if stale:
            self.mode, self._mode_candidate = "DETECT", None
            return
        wanted = ("VIEW" if speed.kmh > self.config.enter_view_kmh else
                  "DETECT" if speed.kmh < self.config.enter_detect_kmh else self.mode)
        if wanted == self.mode:
            self._mode_candidate = None
        else:
            if self._mode_candidate is None or self._mode_candidate[0] != wanted:
                self._mode_candidate = (wanted, now)
            if now - self._mode_candidate[1] >= self.config.mode_dwell_ns:
                self.mode, self._mode_candidate = wanted, None

    def update(self, scene, metadata, speed, now_ns) -> Decision:
        """Invalid inputs hold controls. Stale speed selects a non-brightening DETECT profile."""
        flags = []
        decision_limits = None
        observed_id = observed_sent = None
        def finish(reason, command=None, *, valid=True):
            limits = decision_limits or self._effective(self.config.detect if self.mode == "DETECT" else self.config.view)
            if self._fault:
                flags.append("CONTROL_TIMEOUT")
            return Decision(now_ns if type(now_ns) is int and now_ns >= 0 else None,
                            self.config.version, self.mode, tuple(dict.fromkeys(flags)), reason,
                            limits, command, self._pending.command_id if self._pending else None,
                            self._pending, self._sent_ns,
                            observed_id, observed_sent,
                            metadata if valid else None, scene if valid else None,
                            speed if valid else None)
        try:
            _integer(now_ns, "now_ns")
            if self._last_now is not None and now_ns <= self._last_now:
                raise ValueError("update time must strictly increase")
            if self._sent_ns is not None and now_ns < self._sent_ns:
                raise ValueError("update precedes recorded dispatch time")
        except ValueError as error:
            self._mode_candidate = self._status_candidate = None
            flags.append("INVALID_INPUT")
            return finish(str(error), valid=False)
        self._last_now = now_ns
        if self._pending is not None and now_ns - self._pending.created_ns >= self.config.command_timeout_ns:
            self._fault = True
        try:
            self._validate(scene, metadata, speed, now_ns)
        except ValueError as error:
            self._mode_candidate = self._status_candidate = None
            flags.append("INVALID_INPUT")
            return finish(str(error), valid=False)
        if (now_ns - metadata.exposure_start_ns > self.config.max_frame_age_ns
                or (self._last_frame is not None and metadata.frame_id <= self._last_frame)
                or (self._last_frame_time is not None and metadata.exposure_start_ns <= self._last_frame_time)):
            self._mode_candidate = self._status_candidate = None
            flags.append("STALE_FRAME")
            return finish("hold: old, duplicate or out-of-order frame")
        if (self._last_frame_time is not None
                and metadata.exposure_start_ns - self._last_frame_time > self.config.max_frame_age_ns):
            self._status_candidate = None
        self._last_frame, self._last_frame_time = metadata.frame_id, metadata.exposure_start_ns
        stale_speed = (speed is None or now_ns - speed.timestamp_ns > self.config.max_speed_age_ns
                       or (self._last_speed_time is not None and speed.timestamp_ns < self._last_speed_time))
        if not stale_speed:
            if (self._last_speed_time is not None
                    and speed.timestamp_ns - self._last_speed_time > self.config.max_speed_age_ns):
                self._mode_candidate = None
            self._last_speed_time = speed.timestamp_ns
        self._update_mode(speed, now_ns, stale_speed)
        if stale_speed:
            flags.append("STALE_SPEED")
        if not scene.calibrated:
            flags.append("UNCALIBRATED_ROI")
        highlights = any(zone.clipped_pct >= self.config.highlight_pct for zone in scene.zones)
        if highlights:
            flags.append("HIGHLIGHTS")
        profile = self.config.detect if self.mode == "DETECT" else self.config.view
        limits = self._effective(profile)
        # Fixed cadence constrains actual exposure even if the advertised range is wider.
        if metadata.frame_duration_us < limits.min_exposure_us:
            flags.append("INVALID_INPUT")
            self._status_candidate = self._mode_candidate = None
            return finish("frame duration is shorter than profile minimum", valid=False)
        limits = ExposureLimits(limits.min_exposure_us,
                                min(limits.max_exposure_us, metadata.frame_duration_us),
                                limits.min_gain, limits.max_gain,
                                limits.exposure_tolerance_us, limits.gain_tolerance)
        decision_limits = limits
        if self._fault:
            self._status_candidate = None
            return finish("hold: command timeout latched; verify hardware before reinitializing")
        if self._pending is not None:
            if (self._sent_ns is not None and metadata.frame_id > self._issued_frame_id
                    and metadata.exposure_start_ns > self._sent_ns
                    and self._same_controls(metadata, self._pending)):
                flags.append("COMMAND_OBSERVED")
                observed_id, observed_sent = self._pending.command_id, self._sent_ns
                self._pending, self._sent_ns = None, None
            else:
                self._status_candidate = None
                flags.append("AWAITING_DISPATCH" if self._sent_ns is None else "AWAITING_APPLY")
                if self._pending.exposure_us > limits.max_exposure_us:
                    flags.append("PROFILE_LIMIT_PENDING")
                return finish("hold: one command remains in flight")
        low_y = profile.target_y * (1 - profile.band_fraction)
        high_y = profile.target_y * (1 + profile.band_fraction)
        in_range = (limits.min_exposure_us - limits.exposure_tolerance_us <= metadata.exposure_us
                    <= limits.max_exposure_us + limits.exposure_tolerance_us
                    and limits.min_gain - limits.gain_tolerance <= metadata.analogue_gain
                    <= limits.max_gain + limits.gain_tolerance)
        at_max = (in_range and abs(metadata.exposure_us - limits.max_exposure_us) <= limits.exposure_tolerance_us
                  and abs(metadata.analogue_gain - limits.max_gain) <= limits.gain_tolerance)
        at_min = (in_range and abs(metadata.exposure_us - limits.min_exposure_us) <= limits.exposure_tolerance_us
                  and abs(metadata.analogue_gain - limits.min_gain) <= limits.gain_tolerance)
        candidate = ("LOW_LIGHT" if scene.meter_y < low_y and at_max else
                     "TOO_BRIGHT" if scene.meter_y > high_y and at_min else None)
        if candidate:
            if self._status_candidate is None or self._status_candidate[0] != candidate:
                self._status_candidate = (candidate, now_ns)
            flags.append(candidate if now_ns - self._status_candidate[1] >= self.config.status_dwell_ns
                         else "SATURATION_PENDING")
            return finish("hold: observed control limit prevents reaching brightness band")
        self._status_candidate = None
        if low_y <= scene.meter_y <= high_y and in_range:
            flags.append("NEAR_TARGET")
            return finish("hold: brightness is inside configured band")
        response = min(max(profile.target_y / max(scene.meter_y, 1), self.config.min_response),
                       self.config.max_response)
        if stale_speed or highlights:
            response = min(response, 1.0)
        if response == 1.0 and in_range:
            flags.append("HOLD")
            return finish("hold: stale speed or glare suppresses brightening")
        product = min(max(metadata.exposure_us * metadata.analogue_gain * response,
                          limits.min_exposure_us * limits.min_gain),
                      limits.max_exposure_us * limits.max_gain)
        exposure = min(max(profile.preferred_exposure_us, limits.min_exposure_us), limits.max_exposure_us)
        gain = min(max(product / exposure, limits.min_gain), limits.max_gain)
        exposure = min(max(round(product / gain), limits.min_exposure_us), limits.max_exposure_us)
        command = ExposureCommand(self._next_id, now_ns, int(exposure), float(gain))
        if self._same_controls(metadata, command):
            flags.append("HOLD")
            return finish("hold: requested change is below readback tolerance")
        self._next_id += 1
        self._pending, self._issued_frame_id = command, metadata.frame_id
        self._sent_ns = None
        flags.extend(("ADJUSTING", "AWAITING_DISPATCH"))
        return finish("dispatch required; wait for fresh matching metadata before next command", command)
