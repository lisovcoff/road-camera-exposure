from pathlib import Path
import tempfile
import unittest
import numpy as np
from road_camera_exposure.imaging import RunBudget,visible_y_plane,frame_from_metadata


class ImagingTests(unittest.TestCase):
    def test_stride_and_exposure_timestamp_conversion(self):
        a=np.zeros((18,20),dtype=np.uint8);a[:12,16:]=255
        y=visible_y_plane(a,{'format':'YUV420','size':[16,12],'stride':20})
        self.assertEqual(y.shape,(12,16));self.assertEqual(y.max(),0)
        f=frame_from_metadata({'SensorTimestamp':10_000_000,'ExposureTime':4000,
            'AnalogueGain':2.,'DigitalGain':1.,'FrameDuration':33333})
        self.assertEqual(f.exposure_start_ns,6_000_000)
    def test_budget_reserves_storage_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new';b=RunBudget(out,max_bytes=20,reserve_bytes=100,free_bytes=lambda:200)
            b.write_new('value',b'123')
            with self.assertRaises(FileExistsError):b.write_new('value',b'4')
            with self.assertRaises(ValueError):b.write_new('../escape',b'4')
            with self.assertRaises(ValueError):b.write_new('large',b'x'*21)
            with self.assertRaises(FileExistsError):RunBudget(out,max_bytes=20,reserve_bytes=100,free_bytes=lambda:200)


if __name__=='__main__':unittest.main()
