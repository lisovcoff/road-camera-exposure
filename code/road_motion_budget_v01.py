"""Offline pinhole motion budget. No camera imports, gain policy or dispatch.

Coordinates: camera x right, y down, z forward. Velocity is camera motion,
expressed in the camera frame; scene points are stationary. Local constant
velocity approximation excludes rolling shutter distortion and acceleration.
"""
from dataclasses import dataclass
import math

import numpy as np


def _finite(value, name, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(name+' must be a finite number >= '+str(minimum))


def projected_velocity(points_m, velocity_m_s, omega_rad_s, fx_px, fy_px):
    """Pixel velocity for stationary points under camera translation/rotation.

    Intrinsics must refer to the actual image stream. Distortion should be
    corrected before using this pinhole model, or bounded separately.
    """
    points = np.asarray(points_m, dtype=float)
    velocity = np.asarray(velocity_m_s, dtype=float)
    omega = np.asarray(omega_rad_s, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0
            or velocity.shape != (3,) or omega.shape != (3,)
            or not all(np.isfinite(a).all() for a in (points, velocity, omega))
            or np.any(points[:, 2] <= 0)):
        raise ValueError('finite Nx3 points with positive depth and two 3-vectors required')
    _finite(fx_px, 'fx_px', 1e-12)
    _finite(fy_px, 'fy_px', 1e-12)
    change = -velocity - np.cross(omega, points)
    z = points[:, 2]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        uv = (change[:, :2]*z[:, None] - points[:, :2]*change[:, 2, None]) / z[:, None]**2
        uv *= [fx_px, fy_px]
    if not np.isfinite(uv).all():
        raise ValueError('projection overflow or degenerate depth')
    return uv


@dataclass(frozen=True)
class Budget:
    max_requested_exposure_us: int | None
    max_actual_exposure_us: int
    speed_bound_px_s: float
    blur_limit_px: float
    status: str
    dispatch_permitted: bool = False  # Offline result never authorizes hardware.


def exposure_budget(velocities_px_s, *, blur_limit_px, uncertainty_px_s,
                    now_ns, measured_ns, max_age_ns, frame_duration_us,
                    min_exposure_us, sensor_max_us, readback_tolerance_us,
                    geometry_calibrated):
    """Worst-point bound, not percentile: no hidden loss of near-road coverage.

    Uncertainty is an externally justified bound, including motion variation
    during measurement age/control latency. Subtract readback tolerance so
    requested + tolerance <= predicted actual-exposure bound.
    """
    flow = np.asarray(velocities_px_s, dtype=float)
    if flow.ndim != 2 or flow.shape[1:] != (2,) or not len(flow) or not np.isfinite(flow).all():
        raise ValueError('finite nonempty Nx2 flow required')
    _finite(blur_limit_px, 'blur_limit_px', 1e-12)
    _finite(uncertainty_px_s, 'uncertainty_px_s')
    for name, value, minimum in [('now_ns', now_ns, 0), ('measured_ns', measured_ns, 0),
            ('max_age_ns', max_age_ns, 1), ('frame_duration_us', frame_duration_us, 1),
            ('min_exposure_us', min_exposure_us, 1), ('sensor_max_us', sensor_max_us, 1),
            ('readback_tolerance_us', readback_tolerance_us, 0)]:
        if type(value) is not int or value < minimum:
            raise ValueError(name+' must be an integer >= '+str(minimum))
    if type(geometry_calibrated) is not bool or sensor_max_us < min_exposure_us:
        raise ValueError('invalid calibration flag or exposure interval')
    speed = float(np.max(np.hypot(flow[:, 0], flow[:, 1]))) + uncertainty_px_s
    if not math.isfinite(speed):
        raise ValueError('speed bound overflow')
    if measured_ns > now_ns or now_ns-measured_ns > max_age_ns:
        return Budget(None, 0, speed, blur_limit_px, 'STALE_OR_FUTURE_MOTION')
    cap = min(frame_duration_us, sensor_max_us)
    if speed > 0:
        limit = blur_limit_px/speed*1e6
        cap = min(cap, math.floor(limit)) if limit < cap else cap
    requested = cap-readback_tolerance_us
    if requested < min_exposure_us:
        return Budget(None, cap, speed, blur_limit_px, 'BLUR_CONSTRAINT_INFEASIBLE')
    return Budget(requested, cap, speed, blur_limit_px,
                  'MODEL_BUDGET' if geometry_calibrated else 'UNCALIBRATED_MODEL_ONLY')


def diagnostic_target_cnr(*, electrons_per_ms, exposure_us, read_noise_e,
                          target_contrast, target_width_px, velocity_px_s):
    """Toy linear single-pixel contrast/noise proxy, NOT an IMX708 prediction.

    A 1-D uniform target loses peak contrast by min(1,width/blur). Independent
    target/background noise is approximated by sqrt(2*(signal+read_noise^2)).
    Omits ISP, optics, pixel integration, saturation and detector behavior.
    """
    for name, value, minimum in [('electrons_per_ms', electrons_per_ms, 0),
            ('exposure_us', exposure_us, 1e-12), ('read_noise_e', read_noise_e, 1e-12),
            ('target_width_px', target_width_px, 1e-12), ('velocity_px_s', velocity_px_s, 0),
            ('target_contrast', target_contrast, 0)]:
        _finite(value, name, minimum)
    if target_contrast > 1:
        raise ValueError('target contrast must be <= 1')
    signal = electrons_per_ms*exposure_us/1000
    blur = velocity_px_s*exposure_us/1e6
    contrast = target_contrast*min(1., target_width_px/max(blur, 1e-12))
    return contrast*signal/math.sqrt(2*(signal+read_noise_e**2))
