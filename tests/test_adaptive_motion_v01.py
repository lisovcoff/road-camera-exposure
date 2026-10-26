from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from road_adaptive_motion_v01 import ShutterPolicy,ControlGate,TargetTracker


class AdaptiveTests(unittest.TestCase):
    def test_motion_shortens_shutter_and_stillness_restores_native(self):
        p=ShutterPolicy()
        moving=p.update(200.,True,1_000_000_000)
        self.assertLess(moving['exposure_us'],4000)
        self.assertIsNotNone(p.update(0.,True,1_100_000_000)['exposure_us'])
        self.assertIsNone(p.update(0.,True,2_200_000_000)['exposure_us'])
        self.assertEqual(p.update(None,False,2_300_000_000)['exposure_us'],4000)

    def test_faster_motion_tightens_limit_and_extreme_motion_reports_infeasible(self):
        p=ShutterPolicy();a=p.update(100.,True,1);b=p.update(300.,True,2)
        self.assertLess(b['exposure_us'],a['exposure_us'])
        self.assertIn('infeasible',p.update(5000.,True,3)['reason'])

    def test_gate_requires_fresh_readback_and_never_overlaps_commands(self):
        g=ControlGate();md={'ExposureTime':32680,'SensorTimestamp':1_100_000_000,'ExposureTimeMode':0}
        desired={'exposure_us':4000};cmd=g.update(desired,md,1_200_000_000);g.sent(cmd,1_200_000_001)
        self.assertIsNone(g.update(desired,md,1_300_000_000))
        self.assertIsNone(g.update(desired,{'ExposureTime':3981,'SensorTimestamp':1_400_000_000,'ExposureTimeMode':1},1_410_000_000))
        self.assertEqual(g.observed_count,1)
        cmd=g.update({'exposure_us':None},{'ExposureTime':3981,'SensorTimestamp':1_500_000_000},1_510_000_000)
        g.sent(cmd,1_510_000_001)
        self.assertIsNone(g.update({'exposure_us':None},{'ExposureTime':32680,'SensorTimestamp':1_700_000_000,'ExposureTimeMode':0},1_710_000_000))
        self.assertEqual(g.observed_count,2)

    def test_gate_times_out_unobserved_control(self):
        g=ControlGate();g.sent({'ExposureTime':4000},1)
        with self.assertRaises(TimeoutError):g.update({'exposure_us':4000},{'ExposureTime':32680,'SensorTimestamp':3_000_000_000},3_000_000_001)

    def test_small_tracker_preserves_original_pixel_speed_units(self):
        rng=np.random.default_rng(91)
        image=rng.integers(30,220,(320,580),dtype=np.uint8)
        tracker=TargetTracker()
        self.assertFalse(tracker.observe(image,1_000_000_000)['valid'])
        result=tracker.observe(np.roll(image,8,axis=1),1_100_000_000)
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['speed_px_s'],80.,delta=5.)


if __name__=='__main__':unittest.main()
