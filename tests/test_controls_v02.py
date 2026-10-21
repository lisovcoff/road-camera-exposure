"""Regression for real 2471 -> 2431 us and fractional-gain readback."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import road_pi_controls_v01 as old
import road_pi_controls_v02 as expanded


def rows_for(module, exposure_error, gain_error):
    return [{"step":i,"step_frame":j,"dispatch_started_ns":0,
             "exposure_start_ns":j+1,"now_ns":j+2,
             "actual_metadata":{"ExposureTime":e-exposure_error,"AnalogueGain":g-gain_error}}
            for i,(e,g) in enumerate(module.STEPS) for j in range(module.FRAMES_PER_STEP)]


class ExpandedTests(unittest.TestCase):
    def test_measured_intermediate_error_is_not_hidden(self):
        self.assertEqual(old.characterize(rows_for(old,40,0))["exposure_tolerance_us"],42)
        result=expanded.characterize(rows_for(expanded,40,.2))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["exposure_tolerance_us"],42)
        self.assertAlmostEqual(result["gain_tolerance"],.21)
        self.assertIn((2471,2),expanded.STEPS)

    def test_large_or_unstable_error_still_rejected(self):
        for e,g in ((81,0),(0,.31)):
            result=expanded.characterize(rows_for(expanded,e,g))
            self.assertFalse(result["accepted"])
            self.assertIsNone(result["exposure_tolerance_us"])
        rows=rows_for(expanded,40,0);rows[-1]["actual_metadata"]["ExposureTime"]+=1
        self.assertFalse(expanded.characterize(rows)["accepted"])


if __name__ == "__main__": unittest.main()
