"""Closed-loop integration against delayed fake controls, not hardware."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import road_pi_loop_v02 as loop
import road_pi_controls_v01 as manual
from road_observer_v01 import json_safe
from test_controls_v01 import ManualCamera
from test_observer_v01 import Clock


class LoopTests(unittest.TestCase):
    def calibration(self, camera):
        _, actual = manual.configure_lab(camera)
        return {"schema_version":1,"accepted":True,"camera_model":"imx708_FAKE",
                "actual_camera_configuration":json_safe(actual),"versions":{"fake":True},
                "exposure_tolerance_us":25,"gain_tolerance":.02}

    def execute(self, directory, camera):
        p=Path(directory)/"cal.json"; p.write_text(json.dumps(self.calibration(camera)))
        with patch.object(loop,"package_versions",return_value={"fake":True}), \
             patch.object(loop,"system_metrics",return_value={"fake":True}), \
             patch.object(manual,"package_versions",return_value={"fake":True}), \
             patch.object(manual,"system_metrics",return_value={"fake":True}):
            return loop.run(lambda:camera,p,Path(directory)/"new",camera.clock)

    def test_acknowledged_loop_profiles_and_saturation(self):
        camera=ManualCamera(Clock())
        with tempfile.TemporaryDirectory() as d:
            result=self.execute(d,camera)
            self.assertIsNone(result["error"])
            self.assertGreater(result["commands_sent"],3)
            self.assertEqual(result["commands_sent"],result["commands_observed"])
            rows=[json.loads(s) for s in (Path(d)/"new/observations.jsonl").read_text().splitlines()]
            self.assertEqual({r["decision"]["mode"] for r in rows},{"DETECT","VIEW"})
            self.assertTrue(any("LOW_LIGHT" in r["decision"]["statuses"] for r in rows))
            pending=None
            for r in rows:
                ack=r["decision"]["observed_command_id"]
                if ack is not None:
                    self.assertEqual(ack,pending); pending=None
                if r["command_dispatch"]:
                    self.assertIsNone(pending)
                    pending=r["command_dispatch"]["command_id"]
            self.assertIsNone(pending)
            self.assertEqual(camera.events[-2:],["stop","close"])

    def test_lost_control_stops_without_blind_retry(self):
        class LostCamera(ManualCamera):
            def set_controls(self, values):
                if not self.commands: super().set_controls(values)
                else: self.commands.append(values)
        camera=LostCamera(Clock())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,"CONTROL_TIMEOUT"):
                self.execute(d,camera)
            summary=json.loads((Path(d)/"new/summary.json").read_text())
            self.assertEqual(summary["commands_sent"],1)
            self.assertEqual(summary["commands_observed"],0)
            self.assertEqual(camera.events[-2:],["stop","close"])

    def test_rejects_different_calibration(self):
        camera=ManualCamera(Clock()); cal=self.calibration(camera)
        with patch.object(loop,"package_versions",return_value={"fake":True}):
            actual=camera.camera_configuration()
            self.assertEqual(loop.verified_limits(cal,camera,actual).min_gain,2)
            for key,value in (("accepted",False),("camera_model","other"),
                              ("versions",{}),("exposure_tolerance_us",100)):
                bad=dict(cal); bad[key]=value
                with self.assertRaises(ValueError): loop.verified_limits(bad,camera,actual)


if __name__ == "__main__": unittest.main()
