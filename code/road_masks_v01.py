"""Explicit, uncalibrated exposure-metering polygons; no image/camera I/O.

Normalized coordinates describe image EDGES; rasterization tests pixel centers.
Only convex nondegenerate polygons are supported. These are metering masks,
not an output crop or a claim about physical distance or two-lane coverage.
"""
import math

import numpy as np

from road_core_v02 import ZoneMask


def convex_polygon_mask(shape, points):
    if (not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(type(value) is not int or not 1 <= value <= 4096 for value in shape)):
        raise ValueError("shape must contain height/width integers in [1,4096]")
    if not isinstance(points, (tuple, list)) or not 3 <= len(points) <= 32:
        raise ValueError("polygon requires 3..32 normalized vertices")
    for point in points:
        if (not isinstance(point, (tuple, list)) or len(point) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in point)):
            raise ValueError("polygon coordinates must be finite x/y numbers in [0,1]")
    p = np.asarray(points, dtype=float)
    if len({tuple(point) for point in points}) != len(points):
        raise ValueError("polygon vertices must be unique")
    # For each directed edge, ALL other vertices must lie on the same side.
    # This rejects concave, crossing and degenerate input without a geometry dependency.
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(p, np.roll(p, -1, axis=0)))
    if abs(area) <= 1e-12:
        raise ValueError("polygon has zero area")
    direction = 1 if area > 0 else -1
    height, width = shape
    x = (np.arange(width) + 0.5) / width
    y = ((np.arange(height) + 0.5) / height)[:, None]
    mask = np.ones(shape, dtype=bool)
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        side = (b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0])
        if np.any(direction * side < -1e-12):
            raise ValueError("polygon must be convex and ordered around its boundary")
        cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
        mask &= direction * cross >= -1e-12
    return mask


def build_metering_masks(config, profile_name, shape, *, transform):
    """Return left/center/right zones, excluded pixels and review-only pixels.

    The configuration is deliberately unusable as a calibrated road mask.
    Profiles must be selected explicitly; lighting never selects geometry.
    """
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("mask schema version 1 required")
    if config.get("calibrated") is not False or config.get("scope") != "exposure_metering_only":
        raise ValueError("this builder supports uncalibrated metering candidates only")
    if config.get("coordinates") != "normalized_image_edges":
        raise ValueError("normalized image-edge coordinates required")
    if not isinstance(config.get("version"), str) or not config["version"].strip():
        raise ValueError("mask version required")
    if transform != config.get("transform"):
        raise ValueError("image transform differs from mask coordinates")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("image shape required")
    ratio = config.get("reference_aspect")
    if (not isinstance(ratio, list) or len(ratio) != 2
            or any(type(v) is not int or v < 1 for v in ratio)):
        raise ValueError("reference aspect must be [width,height]")
    if any(type(v) is not int or not 1 <= v <= 4096 for v in shape):
        raise ValueError("invalid image dimensions")
    if shape[1] * ratio[1] != shape[0] * ratio[0]:
        raise ValueError("aspect changed: crop/letterbox requires new coordinates")
    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise ValueError("select a known profile explicitly")
    profile = profiles[profile_name]
    if not isinstance(profile, dict) or "road_polygon" not in profile:
        raise ValueError("profile requires an explicit road polygon")
    road = convex_polygon_mask(shape, profile["road_polygon"])
    exclude = np.zeros(shape, dtype=bool)
    if any(not isinstance(config.get(field, {}), dict) for field in ("exclusions", "review_only")):
        raise ValueError("exclusions/review-only regions must be named polygon maps")
    for points in config.get("exclusions", {}).values():
        exclude |= convex_polygon_mask(shape, points)
    review = np.zeros(shape, dtype=bool)
    for points in config.get("review_only", {}).values():
        review |= convex_polygon_mask(shape, points)
    weights = profile.get("zone_weights")
    if (not isinstance(weights, list) or len(weights) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in weights)):
        raise ValueError("three explicit positive zone weights required")
    # Split the outer polygon by each row's horizontal span BEFORE exclusions;
    # an excluded reflection must not move the remaining zone boundaries.
    rows = road.any(axis=1)
    first = np.argmax(road, axis=1)
    last = shape[1] - 1 - np.argmax(road[:, ::-1], axis=1)
    span = np.maximum(last - first + 1, 1)
    fractions = (np.arange(shape[1])[None, :] - first[:, None] + 0.5) / span[:, None]
    ownership = np.clip(np.floor(fractions * 3), 0, 2).astype(np.int8)
    zones = tuple(ZoneMask(name, road & rows[:, None] & (ownership == number), float(weight))
                  for number, (name, weight) in enumerate(zip(("left", "center", "right"), weights)))
    if any(not np.any(zone.mask & ~exclude) for zone in zones):
        raise ValueError("all three metering zones must retain pixels")
    return zones, exclude, review
