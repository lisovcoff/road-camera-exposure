"""Offline simultaneous sink/control integration, NOT a real-codec test."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import road_pi_encode_v01 as bench
import road_pi_loop_v02 as loop
import road_pi_controls_v01 as manual
from road_observer_v01 import json_safe
from test_encode_v01 import EncodingCamera, FakeEncoder
from test_observer_v01 import Clock


class CombinedTests(unittest.TestCase):
    def test_delayed_control_acknowledgments_with_packet_sink(self):
        camera=EncodingCamera(Clock());_,actual=manual.configure_lab(camera)
        calibration={"schema_version":1,"accepted":True,"camera_model":"imx708_FAKE",
                     "actual_camera_configuration":json_safe(actual),"versions":{"fake":True},
                     "exposure_tolerance_us":42,"gain_tolerance":.23}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"cal.json";p.write_text(json.dumps(calibration))
            with patch.object(bench,"system_metrics",return_value={"throttled":"throttled=0x0"}), \
                 patch.object(manual,"system_metrics",return_value={}), \
                 patch.object(manual,"package_versions",return_value={"fake":True}), \
                 patch.object(loop,"package_versions",return_value={"fake":True}):
                summary=bench.run(lambda:camera,"mjpeg",Path(d)/"new",p,
                    encoder_factory=lambda codec,stats:(FakeEncoder(),stats),
                    decoder=lambda codec,payloads:{"frames_decoded":6,"sizes":[(1920,1080)]},clock=camera.clock)
            self.assertIsNone(summary["error"])
            self.assertGreater(summary["commands_sent"],3)
            self.assertEqual(summary["commands_sent"],summary["commands_observed"])
            self.assertIsNone(summary["pending_command_id"])
            packets=json.loads((Path(d)/"new/packets.json").read_text())
            self.assertGreater(packets["packets"],400)
            self.assertEqual(camera.events[-3:],["encoder_stop","stop","close"])


if __name__ == "__main__": unittest.main()
