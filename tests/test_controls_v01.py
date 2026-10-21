"""Offline delayed readback and calibration rejection tests, no camera import."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import road_pi_controls_v01 as lab
from test_observer_v01 import Clock, FakeCamera, FakeRequest


class ManualRequest(FakeRequest):
    def get_metadata(self):
        md = super().get_metadata()
        md.update(ExposureTime=self.camera.exposure - 19,
                  AnalogueGain=self.camera.gain - .01)
        return md


class ManualCamera(FakeCamera):
    def __init__(self, clock):
        super().__init__(clock)
        self.exposure, self.gain, self.queued, self.commands = 2000, 2., None, []
    def create_video_configuration(self, **kwargs):
        return kwargs
    def capture_request(self, **kwargs):
        assert kwargs == {"wait": False, "flush": False}
        return object()
    def set_controls(self, values):
        self.commands.append(values)
        self.queued = (2, values)
    def wait(self, job, timeout):
        self.clock.now += 33_333_333
        if self.queued:
            count, values = self.queued
            if count == 0:
                self.exposure, self.gain = values["ExposureTime"], values["AnalogueGain"]
                self.queued = None
            else:
                self.queued = count - 1, values
        return ManualRequest(self)


class ControlsTests(unittest.TestCase):
    def run_fake(self, directory, camera=None):
        clock = camera.clock if camera else Clock()
        camera = camera or ManualCamera(clock)
        with patch.object(lab, "system_metrics", return_value={"fake": True}), \
             patch.object(lab, "package_versions", return_value={"fake": True}):
            result = lab.run(lambda: camera, Path(directory) / "new", clock)
        return result, camera

    def test_delayed_controls_measured_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            result, camera = self.run_fake(d)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["samples"], 216)
            self.assertEqual(len(camera.commands), 12)
            cal = json.loads((Path(d) / "new/calibration.json").read_text())
            self.assertEqual(cal["exposure_tolerance_us"], 21)
            self.assertAlmostEqual(cal["gain_tolerance"], .02)
            self.assertIsNone(cal["steps"][0]["first_matching_sample"])
            self.assertTrue(all(s["first_matching_sample"] >= 2 for s in cal["steps"][1:]))
            self.assertEqual(camera.events[-2:], ["stop", "close"])

    def test_missing_or_unstable_tail_rejected(self):
        self.assertFalse(lab.characterize([])["accepted"])
        rows=[]
        for step,(exp,gain) in enumerate(lab.STEPS):
            for i in range(18):
                rows.append({"step":step,"step_frame":i,"dispatch_started_ns":0,
                    "exposure_start_ns":i+1,"now_ns":i+2,
                    "actual_metadata":{"ExposureTime":exp-19,"AnalogueGain":gain}})
        self.assertTrue(lab.characterize(rows)["accepted"])
        bad=copy.deepcopy(rows); bad[-1]["actual_metadata"]["ExposureTime"] += 1
        self.assertFalse(lab.characterize(bad)["accepted"])
        bad=copy.deepcopy(rows)
        for r in bad: r["exposure_start_ns"] = -1
        self.assertFalse(lab.characterize(bad)["accepted"])
        bad=copy.deepcopy(rows)
        for r in bad: r["actual_metadata"]["AnalogueGain"] += 1
        self.assertFalse(lab.characterize(bad)["accepted"])

    def test_capture_error_releases_and_closes(self):
        camera=ManualCamera(Clock())
        with tempfile.TemporaryDirectory() as d, \
             patch.object(ManualRequest,"make_array",side_effect=ValueError("bad array")):
            with self.assertRaisesRegex(ValueError,"bad array"):
                self.run_fake(d,camera)
            self.assertIn("release",camera.events)
            self.assertEqual(camera.events[-2:],["stop","close"])
            self.assertIn("bad array",json.loads((Path(d)/"new/summary.json").read_text())["error"])


if __name__ == "__main__":
    unittest.main()
