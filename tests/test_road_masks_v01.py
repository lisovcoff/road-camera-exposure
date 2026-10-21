"""Coordinate and exclusion checks; no road-data or Pillow dependency."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from road_masks_v01 import build_metering_masks, convex_polygon_mask
from road_core_v02 import AnalysisConfig, analyze_y_plane


class PolygonTests(unittest.TestCase):
    def test_full_rectangle_both_windings_includes_every_pixel_center(self):
        square = [[0, 0], [1, 0], [1, 1], [0, 1]]
        for points in (square, list(reversed(square))):
            self.assertTrue(convex_polygon_mask((12, 16), points).all())

    def test_triangle_pixel_centers_and_diagonal_boundary(self):
        mask = convex_polygon_mask((4, 4), [[0, 0], [1, 0], [0, 1]])
        y, x = np.indices((4, 4))
        np.testing.assert_array_equal(mask, x + y <= 3)

    def test_rejects_crossing_concave_degenerate_and_invalid_vertices(self):
        variants = (
            [[0, 0], [1, 1], [0, 1], [1, 0]],
            [[0, 0], [1, 0], [0.4, 0.4], [1, 1], [0, 1]],
            [[0, 0], [0.5, 0.5], [1, 1]],
            [[0, 0], [1, 0], [1, 0], [0, 1]],
            [[0, 0], [float("nan"), 0], [1, 1]],
            [[0, 0], [1.1, 0], [1, 1]],
            [[False, 0], [1, 0], [1, 1]],
        )
        for points in variants:
            with self.subTest(points=points), self.assertRaises(ValueError):
                convex_polygon_mask((12, 16), points)

    def test_rejects_unbounded_or_invalid_dimensions(self):
        for shape in ((0, 16), (12.0, 16), (12, True), (5000, 5000), (16,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                convex_polygon_mask(shape, [[0, 0], [1, 0], [1, 1]])


class CandidateMaskTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs" / "road_roi_candidate_v01.json").read_text(encoding="utf-8"))

    def test_every_profile_produces_disjoint_nonempty_zones(self):
        for profile, settings in self.config["profiles"].items():
            zones, exclude, _ = build_metering_masks(self.config, profile, (240, 320), transform="identity")
            combined = np.stack([zone.mask for zone in zones]).sum(axis=0)
            self.assertLessEqual(int(combined.max()), 1)
            np.testing.assert_array_equal(combined.astype(bool),
                convex_polygon_mask((240, 320), settings["road_polygon"]))
            self.assertTrue(all((zone.mask & ~exclude).any() for zone in zones))

    def test_exclusion_does_not_shift_zone_ownership(self):
        config = deepcopy(self.config)
        profile = "night_road_reference"
        original, _, _ = build_metering_masks(config, profile, (240, 320), transform="identity")
        config["exclusions"]["test_hole"] = [[0.4, 0.5], [0.6, 0.5], [0.6, 0.6], [0.4, 0.6]]
        modified, exclude, _ = build_metering_masks(config, profile, (240, 320), transform="identity")
        for before, after in zip(original, modified):
            np.testing.assert_array_equal(before.mask, after.mask)
        self.assertTrue(np.any(exclude & original[1].mask))

    def test_unknown_artifact_remains_review_only_not_excluded(self):
        zones, exclude, review = build_metering_masks(self.config, "night_road_reference", (240, 320), transform="identity")
        selected = np.logical_or.reduce([zone.mask for zone in zones])
        self.assertTrue(np.any(selected & review))
        self.assertFalse(np.any(selected & review & exclude))

    def test_dark_road_is_kept_and_can_be_analyzed_by_v02(self):
        zones, exclude, _ = build_metering_masks(self.config, "left_road_reference", (240, 320), transform="identity")
        scene = analyze_y_plane(np.zeros((240, 320), dtype=np.uint8), zones,
            AnalysisConfig(32, 235, 50, 16), exclude=exclude,
            mask_version=self.config["version"], transform="identity")
        self.assertEqual(scene.meter_y, 0)
        self.assertAlmostEqual(scene.dark_pct, 100)
        self.assertFalse(scene.calibrated)

    def test_wrong_transform_crop_and_unknown_profile_are_rejected(self):
        for profile, shape, transform in (("night_road_reference", (180, 320), "identity"),
                                           ("night_road_reference", (240, 320), "rotate180"),
                                           ("missing", (240, 320), "identity")):
            with self.subTest(profile=profile, shape=shape), self.assertRaises(ValueError):
                build_metering_masks(self.config, profile, shape, transform=transform)

    def test_cannot_silently_claim_metric_calibration_or_use_invalid_weights(self):
        for change in ("calibration", "weight"):
            config = deepcopy(self.config)
            if change == "calibration":
                config["calibrated"] = True
            else:
                config["profiles"]["night_road_reference"]["zone_weights"][0] = 0
            with self.assertRaises(ValueError):
                build_metering_masks(config, "night_road_reference", (240, 320), transform="identity")


if __name__ == "__main__":
    unittest.main()
