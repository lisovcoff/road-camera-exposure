from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from road_colour_preview_v01 import preview_planes


class ColourPreviewTests(unittest.TestCase):
    def test_chroma_planes_stride_crop_rotate_without_changing_luma(self):
        a=np.full((18,20),255,dtype=np.uint8)
        a[:12,:16]=np.arange(192,dtype=np.uint8).reshape(12,16)
        packed=a.reshape(36,10);packed[24:30,:8]=40;packed[30:36,:8]=200
        before=a.copy();y,u,v=preview_planes(a,{'format':'YUV420','size':(16,12),'stride':20},[2,2,14,10])
        self.assertEqual(y.shape,(8,12));self.assertEqual(u.shape,(4,6));self.assertEqual(v.shape,(4,6))
        np.testing.assert_array_equal(y,a[2:10,2:14][::-1,::-1])
        self.assertTrue((u==40).all());self.assertTrue((v==200).all())
        np.testing.assert_array_equal(a,before)

    def test_odd_or_out_of_bounds_crop_rejected(self):
        a=np.zeros((18,20),dtype=np.uint8);cfg={'format':'YUV420','size':(16,12),'stride':20}
        for roi in ([1,0,12,10],[0,0,18,10]):
            with self.assertRaises(ValueError):preview_planes(a,cfg,roi)


if __name__=='__main__':unittest.main()
