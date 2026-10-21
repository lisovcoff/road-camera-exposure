"""Bounded packet sink and fake encoder lifecycle; no real codec claims."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import road_pi_encode_v01 as bench
import road_pi_controls_v01 as manual
from test_controls_v01 import ManualCamera
from test_observer_v01 import Clock


class FakeEncoder:
    firsttimestamp=None


class EncodingCamera(ManualCamera):
    def start_encoder(self,encoder,sink,name):
        assert name=="main"
        self.encoder,self.sink=encoder,sink
        self.events.append("encoder_start")
    def stop_encoder(self,encoder):
        self.events.append("encoder_stop")
    def wait(self,job,timeout):
        request=super().wait(job,timeout)
        if hasattr(self,"encoder"):
            timestamp=request.get_metadata()["SensorTimestamp"]//1000
            if self.encoder.firsttimestamp is None: self.encoder.firsttimestamp=timestamp
            self.sink.add(b"FAKE",timestamp-self.encoder.firsttimestamp,True,self.clock.now)
        return request


class EncoderTests(unittest.TestCase):
    def test_sink_limits_and_clock_order_are_latched(self):
        sink=bench.PacketStats(max_packets=2,max_bytes=10)
        sink.add(b"123",0,True,10);sink.add(b"456",33333,False,40)
        self.assertEqual(sink.result(0)["packets"],2)
        sink.add(b"7",66666,False,70)
        with self.assertRaises(RuntimeError): sink.check()
        self.assertEqual(len(sink.rows),2)
        for stamp in (None,-1):
            sink=bench.PacketStats();sink.add(b"x",stamp,True,0)
            with self.assertRaises(RuntimeError): sink.check()
        sink=bench.PacketStats();sink.add(b"x",0,True,0);sink.add(b"x",0,True,1)
        with self.assertRaises(RuntimeError): sink.check()

    def test_payload_memory_bound(self):
        sink=bench.PacketStats()
        for i in range(12):sink.add(b"x"*700_000,i,True,i)
        self.assertLessEqual(sink.retained,3_000_000)
        self.assertLessEqual(len(sink.payloads),6)

    def execute(self,d,decoder):
        camera=EncodingCamera(Clock())
        with patch.object(bench,"system_metrics",return_value={"throttled":"throttled=0x0"}), \
             patch.object(manual,"system_metrics",return_value={}), \
             patch.object(manual,"package_versions",return_value={}):
            summary=bench.run(lambda:camera,"h264",Path(d)/"new",
                encoder_factory=lambda codec,stats:(FakeEncoder(),stats),decoder=decoder,clock=camera.clock)
        return summary,camera

    def test_fake_encode_runs_capture_and_closes(self):
        with tempfile.TemporaryDirectory() as d:
            result,camera=self.execute(d,lambda c,p:{"frames_decoded":6,"sizes":[(1920,1080)]})
            self.assertIsNone(result["error"])
            self.assertEqual(camera.events[-3:],["encoder_stop","stop","close"])
            packets=json.loads((Path(d)/"new/packets.json").read_text())
            self.assertGreater(packets["packets"],200)
            self.assertEqual(result["commands_sent"],0)

    def test_decode_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,"experiment failed"):
                self.execute(d,lambda c,p:{"frames_decoded":0,"sizes":[]})
            result=json.loads((Path(d)/"new/summary.json").read_text())
            self.assertIn("decode",result["error"])


if __name__ == "__main__": unittest.main()
