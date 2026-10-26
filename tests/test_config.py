import copy
import json
from pathlib import Path
import tempfile
import unittest

from road_camera_exposure.config import load,validate,runtime_args
from road_camera_exposure.cli import main
from road_camera_exposure.adapters.base import exposure_to_device_units,CapturedFrame
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def setUp(self):self.c=load(ROOT/'configs/module3.example.json')
    def test_example_is_valid_but_cannot_control_without_reference(self):
        self.assertIsNone(self.c['colour_gains'])
        with self.assertRaisesRegex(ValueError,'reference'):runtime_args(self.c,'unused')
    def test_measured_configuration_creates_independent_runtime(self):
        self.c['colour_gains']=[1.8,2.0]
        args=runtime_args(self.c,'new-run',bind='192.168.1.50')
        self.assertEqual(args.bind,'192.168.1.50');self.assertEqual(self.c['bind'],'127.0.0.1')
        self.assertEqual(args.output,Path('new-run'))
    def test_invalid_network_crop_and_driver_rejected(self):
        for change in ({'bind':'0.0.0.0'},{'bind':'8.8.8.8'},{'driver':'uvc'},
                       {'roi':[1,0,640,360]},{'tracking_rectangle':[0,0,800,500]},
                       {'colour_gains':[float('nan'),1.]},{'reserve_bytes':0}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate({**self.c,**change})
    def test_dry_run_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new'
            self.assertEqual(main(['run','--config',str(ROOT/'configs/module3.example.json'),'--output',str(out)]),0)
            self.assertFalse(out.exists())
    def test_device_units_and_midpoint_are_explicit(self):
        self.assertEqual(exposure_to_device_units(3981,100),39)
        with self.assertRaises(ValueError):exposure_to_device_units(20,100)
        f=CapturedFrame(np.zeros((8,8),dtype=np.uint8),1_000_000,4000,2.,1.,33333)
        self.assertEqual(f.midpoint_ns,3_000_000)


if __name__=='__main__':unittest.main()
