"""Portable, camera-independent brightness analysis and exposure controller.

Experiment only. The ROI and thresholds are provisional and NOT calibrated
for the road. Exposure is in microseconds; gain is a linear multiplier.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RoiStats:
    median: float
    p10: float
    p90: float
    p99: float
    dark_pct: float
    clipped_pct: float


def analyze_y_plane(y_plane):
    """Analyze temporary centered 160x90 ROI in a 320x180 Y plane."""
    if y_plane.shape[0] < 180 or y_plane.shape[1] < 320:
        raise ValueError(f"Expected at least 180x320 luminance, got {y_plane.shape}")
    roi = y_plane[45:135, 80:240]
    return RoiStats(
        median=float(np.median(roi)),
        p10=float(np.percentile(roi, 10)),
        p90=float(np.percentile(roi, 90)),
        p99=float(np.percentile(roi, 99)),
        dark_pct=float(np.mean(roi <= 32) * 100),
        clipped_pct=float(np.mean(roi >= 235) * 100),
    )


def clamp(value, low, high):
    return min(max(value, low), high)


class ExposureController:
    """Experimental 2 Hz proportional controller, independent of camera APIs."""

    def __init__(self, min_gain=1.2, max_gain=16.0):
        self.mode = "DETECT"
        self.pending_mode = None
        self.pending_since = None
        self.min_gain = min_gain
        self.max_gain = max_gain

    def _update_profile(self, speed_kmh, now_seconds):
        # Hysteresis: switch up only above 11, down only below 9 km/h.
        desired = ("VIEW" if speed_kmh > 11.0 else
                   "DETECT" if speed_kmh < 9.0 else self.mode)
        if desired == self.mode:
            self.pending_mode = None
            self.pending_since = None
        elif self.pending_mode != desired:
            self.pending_mode = desired
            self.pending_since = now_seconds
        elif now_seconds - self.pending_since >= 2.0:
            self.mode = desired
            self.pending_mode = None
            self.pending_since = None

    def decide(self, stats, actual_exposure_us, actual_gain, speed_kmh, now_seconds):
        self._update_profile(speed_kmh, now_seconds)
        detect = self.mode == "DETECT"
        minimum_exp = 2000
        max_exp = 4000 if detect else 12000
        preferred_exp = 4000 if detect else 8000
        target_median = 100.0 if detect else 105.0

        # Conservative response to an arbitrary provisional ROI brightness.
        response = clamp(target_median / max(stats.median, 1.0), 0.80, 1.25)
        if stats.clipped_pct >= 1.0 and stats.median >= 70:
            response = min(response, 0.85)

        current_product = max(float(actual_exposure_us), 1.0) * max(float(actual_gain), 1.0)
        requested_product = current_product * response

        # Prefer 4 ms in DETECT, 8 ms in VIEW; regulate gain first.
        exposure = float(preferred_exp)
        gain = requested_product / exposure
        if gain < self.min_gain:
            gain = self.min_gain
            exposure = requested_product / gain
        elif gain > self.max_gain:
            gain = self.max_gain
            exposure = requested_product / gain

        exposure = int(round(clamp(exposure, minimum_exp, max_exp)))
        gain = round(clamp(gain, self.min_gain, self.max_gain), 3)

        status = "ADJUSTING"
        if (stats.median < 80 and
                actual_exposure_us * actual_gain >= max_exp * self.max_gain * 0.97 and
                response > 1.0):
            status = "LOW_LIGHT"
        elif (stats.median > target_median * 1.25 and
              actual_exposure_us * actual_gain <= minimum_exp * self.min_gain * 1.05):
            status = "TOO_BRIGHT"
        elif stats.clipped_pct >= 1.0:
            status = "HIGHLIGHTS"
        elif target_median * 0.85 <= stats.median <= target_median * 1.15:
            status = "NEAR_TARGET"

        return {
            "mode": self.mode,
            "status": status,
            "requested_exposure_us": exposure,
            "requested_analogue_gain": gain,
            "target_median": target_median,
            "response_factor": round(response, 3),
            "limits_exposure_us": [minimum_exp, max_exp],
        }
