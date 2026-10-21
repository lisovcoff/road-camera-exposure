"""Camera-free shadow/adapter lifecycle tests. No hardware claims."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "tools"))
from road_core_v02 import FrameMetadata, SpeedSample
from road_observer_v01 import RunBudget, ShadowAdvisor, frame_from_metadata, visible_y_plane
import road_pi_observe_v01 as adapter
from simulate_v02_001 import NS, synthetic_configuration


class Clock:
    def __init__(self):
        self.now = 10 * NS
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += round(seconds * NS)


class FakeRequest:
    def __init__(self, camera):
        self.camera = camera
    def get_metadata(self):
        return {"SensorTimestamp": self.camera.clock.now - 1_000_000,
                "ExposureTime": 4000, "AnalogueGain": 8.0, "DigitalGain": 1.0,
                "FrameDuration": 33333, "LensPosition": 0.3}
    def make_array(self, stream):
        assert stream == "lores"
        return np.full((270, 384), 50, dtype=np.uint8)
    def save(self, stream, buffer, format):
        assert stream == "main" and format == "jpeg"
        buffer.write(b"MOCK-JPEG-NOT-A-REAL-PHOTOGRAPH")
    def release(self):
        self.camera.events.append("release")


class FakeCamera:
    camera_controls = {"ExposureTime": (13, 112015096, 20000), "AnalogueGain": (1.1228, 16.0, 1.0)}
    camera_properties = {"Model": "imx708_FAKE"}
    def __init__(self, clock):
        self.clock, self.events = clock, []
        self.actual = {"main": {"size": (1920, 1080), "format": "YUV420", "stride": 1920},
                       "lores": {"size": (320, 180), "format": "YUV420", "stride": 384},
                       "sensor": {"output_size": (2304, 1296), "bit_depth": 10},
                       "transform": "identity"}
    def create_video_configuration(self, **kwargs):
        self.events.append("create_configuration")
        assert kwargs["controls"] == {"FrameRate": 30}
        return kwargs
    def configure(self, config):
        self.events.append("configure")
    def camera_configuration(self):
        return self.actual
    def start(self):
        self.events.append("start")
    def capture_request(self, **kwargs):
        assert kwargs == {"wait": False, "flush": True}
        self.events.append("capture_request")
        return object()
    def wait(self, job, timeout):
        assert timeout == 2
        self.clock.now += 33_333_333
        return FakeRequest(self)
    def set_controls(self, *args, **kwargs):
        raise AssertionError("observer tried to send controls")
    def stop(self):
        self.events.append("stop")
    def close(self):
        self.events.append("close")


class ShadowTests(unittest.TestCase):
    def setUp(self):
        hardware, config, _ = synthetic_configuration()
        self.advisor = ShadowAdvisor(hardware, config)
        zones = adapter.laboratory_zones(adapter.load_config(ROOT / "configs/pi_observer_lab_v01.json"))
        from road_core_v02 import analyze_y_plane, AnalysisConfig
        self.scene = analyze_y_plane(np.full((180, 320), 50, dtype=np.uint8), zones,
                                    AnalysisConfig(32, 235, 50, 16), mask_version="test", transform="identity")

    def test_unapplied_advice_never_accumulates_or_times_out(self):
        proposed = []
        for i in range(30):
            now = NS + i * 100_000_000
            record = self.advisor.observe(self.scene, FrameMetadata(i, now - 10_000_000, 4000, 8, 1, 33333),
                                          SpeedSample(8, now), now)
            proposed.append(record["would_request"])
            self.assertFalse(record["command_dispatched"])
            self.assertIsNone(record["command"])
            self.assertIsNone(record["pending_command"])
            self.assertNotIn("COMMAND_OBSERVED", record["statuses"])
            self.assertNotIn("CONTROL_TIMEOUT", record["statuses"])
        self.assertTrue(all(item == proposed[0] for item in proposed))
        self.assertIsNotNone(proposed[0])

    def test_mode_dwell_survives_discarding_proposals(self):
        for i in range(9):
            now = NS + i * (NS // 4)
            record = self.advisor.observe(self.scene, FrameMetadata(i, now - 10_000_000, 4000, 8, 1, 33333),
                                          SpeedSample(13, now), now)
        self.assertEqual(record["mode"], "VIEW")


class InputTests(unittest.TestCase):
    def test_stride_and_uv_padding_are_removed(self):
        array = np.full((270, 384), 255, dtype=np.uint8)
        array[:180, :320] = 82
        plane = visible_y_plane(array, {"size": (320, 180), "stride": 384, "format": "YUV420"})
        self.assertEqual(plane.shape, (180, 320))
        self.assertTrue(np.all(plane == 82))

    def test_wrong_stride_shape_format_and_dtype_are_rejected(self):
        config = {"size": (320, 180), "stride": 384, "format": "YUV420"}
        for array, actual in ((np.zeros((270, 320), dtype=np.uint8), config),
                              (np.zeros((270, 384)), config),
                              (np.zeros((270, 384), dtype=np.uint8), dict(config, format="RGB888"))):
            with self.assertRaises(ValueError):
                visible_y_plane(array, actual)

    def test_timestamp_units_and_raw_id_are_not_invented(self):
        md = {"SensorTimestamp": NS, "ExposureTime": 4000, "FrameDuration": 33333,
              "AnalogueGain": 8.0, "DigitalGain": 1.0}
        frame = frame_from_metadata(md)
        self.assertEqual(frame.exposure_start_ns, NS - 4_000_000)
        self.assertEqual(frame.frame_id, NS)
        for bad in ({}, dict(md, ExposureTime=4.0), dict(md, SensorTimestamp=1), dict(md, DigitalGain=float("nan"))):
            with self.assertRaises(ValueError):
                frame_from_metadata(bad)

    def test_legacy_geometry_and_unbounded_configs_are_rejected(self):
        valid = json.loads((ROOT / "configs/pi_observer_lab_v01.json").read_text())
        for changes in ({"duration_s": 100}, {"max_output_bytes": 100_000_000},
                        {"reserve_bytes": 1}, {"lores_size": [320, 240]},
                        {"roi_source": "dashcam_dataset"}, {"snapshot_at_s": [0, 1, 2, 3]}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.json"
                path.write_text(json.dumps(dict(valid, **changes)))
                with self.assertRaises(ValueError):
                    adapter.load_config(path)


class BudgetTests(unittest.TestCase):
    def test_output_is_new_and_budget_and_reserve_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            free = [1000]
            budget = RunBudget(path, max_bytes=10, reserve_bytes=100, free_bytes=lambda: free[0])
            budget.write_new("one.bin", b"12345")
            with self.assertRaises(FileExistsError):
                budget.write_new("one.bin", b"X")
            self.assertEqual((path / "one.bin").read_bytes(), b"12345")
            with self.assertRaises(ValueError):
                budget.write_new("two.bin", b"123456")
            free[0] = 100
            with self.assertRaises(ValueError):
                budget.write_new("three.bin", b"1")


class AdapterTests(unittest.TestCase):
    def config(self):
        config = adapter.load_config(ROOT / "configs/pi_observer_lab_v01.json")
        return dict(config, duration_s=1, snapshot_at_s=[0])

    def run_fake(self, camera, clock, output, config=None):
        with patch.object(adapter, "package_versions", return_value={"packages": "FAKE"}):
            return adapter.run_observation(lambda: camera, config or self.config(), output,
                clock=clock, sleep=clock.sleep, metrics=lambda: {"throttled": "FAKE"},
                budget_factory=lambda path, **kw: RunBudget(path, free_bytes=lambda: 10**12, **kw))

    def test_default_and_missing_run_arguments_never_import_camera(self):
        with patch.dict(sys.modules, {"picamera2": None}), patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(adapter.main([]), 0)
        with patch.dict(sys.modules, {"picamera2": None}), patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(adapter.main(["--run-camera"]), 2)

    def test_coupled_request_logging_no_controls_and_cleanup(self):
        clock = Clock();camera = FakeCamera(clock)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = self.run_fake(camera, clock, output)
            self.assertEqual(result["samples"], 2)
            self.assertEqual(result["command_dispatch_count"], 0)
            rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["camera_controls_sent_this_sample"] == {} for row in rows))
            self.assertTrue(all(row["statistics"]["calibrated"] is False for row in rows))
            self.assertTrue(all(row["would_request"] is not None for row in rows))
            self.assertEqual(len(list(output.glob('*.jpg'))), 1)
            self.assertEqual(camera.events.count("release"), 2)
            self.assertEqual(camera.events[-2:], ["stop", "close"])

    def test_timeout_releases_no_phantom_request_and_closes_camera(self):
        clock = Clock();camera = FakeCamera(clock)
        camera.wait = lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("FAKE"))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(TimeoutError):
            self.run_fake(camera, clock, Path(directory) / "run")
        self.assertEqual(camera.events.count("release"), 0)
        self.assertEqual(camera.events[-2:], ["stop", "close"])

    def test_request_failure_and_snapshot_limit_still_release_and_close(self):
        for failure_kind in ("array", "snapshot"):
            clock = Clock();camera = FakeCamera(clock)
            if failure_kind == "array":
                with patch.object(FakeRequest, "make_array", side_effect=RuntimeError("FAKE")):
                    with tempfile.TemporaryDirectory() as directory, self.assertRaises(RuntimeError):
                        self.run_fake(camera, clock, Path(directory) / "run")
            else:
                config = self.config();config["snapshot_max_bytes"] = 1
                with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                    self.run_fake(camera, clock, Path(directory) / "run", config)
            self.assertEqual(camera.events.count("release"), 1)
            self.assertEqual(camera.events[-2:], ["stop", "close"])

    def test_actual_size_mismatch_aborts_before_start(self):
        clock = Clock();camera = FakeCamera(clock)
        camera.actual["lores"]["size"] = (320, 240)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            self.run_fake(camera, clock, Path(directory) / "run")
        self.assertNotIn("start", camera.events)
        self.assertEqual(camera.events[-2:], ["stop", "close"])

    def test_stop_failure_still_attempts_close(self):
        clock = Clock();camera = FakeCamera(clock)
        def failed_stop():
            camera.events.append("stop")
            raise RuntimeError("FAKE STOP")
        camera.stop = failed_stop
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaises(RuntimeError):
                self.run_fake(camera, clock, output)
            summary = json.loads((output / "summary.json").read_text())
            self.assertIn("stop: RuntimeError", summary["cleanup_error"])
        self.assertEqual(camera.events[-2:], ["stop", "close"])

    def test_existing_output_prevents_camera_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                adapter.run_observation(lambda: self.fail("camera should not be constructed"),
                    self.config(), directory, metrics=lambda: {},
                    budget_factory=lambda path, **kw: RunBudget(path, free_bytes=lambda: 10**12, **kw))


if __name__ == "__main__":
    unittest.main()
