from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from road_pi_live_v03 import initial_controls


class NativeControlsTests(unittest.TestCase):
    def test_native_clears_manual_values_and_retains_fixed_focus(self):
        args = SimpleNamespace(adaptation='native', lens=.35,
                               initial_exposure=8951, initial_gain=16)
        controls = initial_controls(args)
        self.assertTrue(controls['AeEnable'])
        for key in ('ExposureTime', 'AnalogueGain', 'ExposureTimeMode', 'AnalogueGainMode'):
            self.assertEqual(controls[key], 0)
        self.assertEqual(controls['LensPosition'], .35)
        self.assertEqual(controls['AfMode'], 0)
        self.assertEqual(controls['ExposureValue'], 0)

    def test_custom_and_fixed_keep_identical_initial_exposure(self):
        controls = [initial_controls(SimpleNamespace(adaptation=mode, lens=0.,
                    initial_exposure=8951, initial_gain=16)) for mode in ('auto', 'fixed')]
        self.assertEqual(controls[0], controls[1])
        self.assertFalse(controls[0]['AeEnable'])
        self.assertEqual(controls[0]['ExposureTime'], 8951)


if __name__ == '__main__':
    unittest.main()
