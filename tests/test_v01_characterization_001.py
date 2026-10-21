"""Baseline evidence and explicit expected failures for known v0.1 defects."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from audit_v01_001 import audit, read_records, replay
from road_core_v01 import ExposureController, RoiStats, analyze_y_plane

FIXTURES = ROOT / "tests" / "fixtures" / "v01"


def stats(value):
    return RoiStats(value, value, value, value, 0.0, 0.0)


class RecordedEvidence(unittest.TestCase):
    def test_fixture_provenance_and_row_counts(self):
        manifest = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
        for name, source in manifest["sources"].items():
            with self.subTest(dataset=name):
                path = FIXTURES / f"{name}.jsonl"
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source["fixture_sha256"])
                self.assertEqual(len(read_records(path)), source["source_rows"])
                self.assertEqual(source["source_rows"], 48)

    def test_lamp_feedback_replays_exactly(self):
        report = audit(FIXTURES / "light_response.jsonl")
        self.assertEqual(report["exact_replay_matches"], 48)
        self.assertEqual(report["actual_exposure_us_range"], [3981, 3981])

    def test_mode_feedback_preserves_known_timestamp_discrepancy(self):
        rows = read_records(FIXTURES / "auto_controller.jsonl")
        mismatches = replay(rows)
        self.assertEqual([item["line"] for item in mismatches], [17])
        self.assertEqual(mismatches[0]["recorded"]["mode"], "VIEW")
        self.assertEqual(mismatches[0]["replayed"]["mode"], "DETECT")
        self.assertEqual(replay(rows, boundary_probe=True), [])

    def test_audit_rejects_missing_nonfinite_and_unordered_feedback(self):
        rows = read_records(FIXTURES / "light_response.jsonl")
        variants = [[], [{"elapsed_s": 0}], [rows[1], rows[0]],
                    [dict(rows[0], elapsed_s=float("nan"))]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            for data in variants:
                with self.subTest(data=data[:1]):
                    path.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_records(path)


class ControllerCharacterization(unittest.TestCase):
    def test_padding_and_non_roi_pixels_do_not_change_statistics(self):
        plane = np.full((270, 384), 255, dtype=np.uint8)
        plane[45:135, 80:240] = 100
        result = analyze_y_plane(plane[:180, :320])
        self.assertEqual(result, stats(100))

    def test_hysteresis_requires_continuous_dwell_and_resets(self):
        controller = ExposureController()
        def mode(speed, time):
            return controller.decide(stats(100), 3981, 8, speed, time)["mode"]
        self.assertEqual(mode(13, 0), "DETECT")
        self.assertEqual(mode(10, 1), "DETECT")
        self.assertEqual(mode(13, 2), "DETECT")
        self.assertEqual(mode(13, 3.9), "DETECT")
        self.assertEqual(mode(13, 4), "VIEW")
        self.assertEqual(mode(9, 5), "VIEW")
        self.assertEqual(mode(8, 6), "VIEW")
        self.assertEqual(mode(8, 8), "DETECT")

    def test_requests_remain_bounded_across_dark_bright_and_mode_changes(self):
        controller = ExposureController()
        for time in range(20):
            for median in (0, 22, 79, 80, 84, 100, 200, 255):
                decision = controller.decide(stats(median), 3981, 16, 13 if time < 10 else 8, time)
                lo, hi = decision["limits_exposure_us"]
                self.assertLessEqual(lo, decision["requested_exposure_us"])
                self.assertLessEqual(decision["requested_exposure_us"], hi)
                self.assertLessEqual(1.2, decision["requested_analogue_gain"])
                self.assertLessEqual(decision["requested_analogue_gain"], 16)

    @unittest.expectedFailure
    def test_known_defect_saturated_dark_scene_should_report_low_light(self):
        self.assertEqual(ExposureController().decide(stats(82), 3981, 16, 8, 0)["status"], "LOW_LIGHT")

    @unittest.expectedFailure
    def test_known_defect_rgb_is_not_a_y_plane(self):
        with self.assertRaises(ValueError):
            analyze_y_plane(np.full((180, 320, 3), 100, dtype=np.uint8))

    @unittest.expectedFailure
    def test_known_defect_nan_pixels_should_be_rejected(self):
        with self.assertRaises(ValueError):
            analyze_y_plane(np.full((180, 320), np.nan))

    @unittest.expectedFailure
    def test_known_defect_invalid_speed_should_be_rejected(self):
        with self.assertRaises(ValueError):
            ExposureController().decide(stats(100), 3981, 8, float("nan"), 0)


if __name__ == "__main__":
    unittest.main()
