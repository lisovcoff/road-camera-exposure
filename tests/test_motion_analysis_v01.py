from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from road_motion_analysis_v01 import sample,smooth3,register_affine,feature_profile,matched_velocity_rows


class MotionAnalysisTests(unittest.TestCase):
    def test_affine_rotation_scale_translation_and_photometric_change(self):
        rng=np.random.default_rng(22)
        reference=smooth3(rng.uniform(20,180,(128,128)))
        angle=.05;scale=1.02
        matrix=scale*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        center=np.array([60.,60.]);translation=center+[3.,-2.]
        yy,xx=np.mgrid[:128,:128]
        coords=np.linalg.inv(matrix)@(np.stack((xx.ravel(),yy.ravel()))-translation[:,None])+center[:,None]
        image,_=sample(reference,coords[0],coords[1]);image=image.reshape(reference.shape)*1.2+7
        fit=register_affine(reference,image,(30,30,90,90))
        self.assertTrue(fit['accepted'])
        np.testing.assert_allclose(fit['warp'][:,:2],matrix,atol=.02)
        np.testing.assert_allclose(fit['warp'][:,2],translation,atol=.3)

    def test_blurred_line_is_wider_than_same_static_reference(self):
        ref=np.full((128,128),150.);ref[:,59:61]=30
        blur=sum(np.roll(ref,i,axis=1) for i in range(-3,4))/7
        warp=np.array([[1.,0.,60.],[0.,1.,60.]])
        args=dict(rectangle=(20,20,100,100),line_window=(45,75),y_range=(30,80))
        a=feature_profile(ref,ref,warp,**args);b=feature_profile(ref,blur,warp,**args)
        self.assertTrue(a['valid']);self.assertTrue(b['valid'])
        self.assertGreater(b['width_fwhm_px'],a['width_fwhm_px']*2)
        self.assertLess(b['relative_contrast'],a['relative_contrast'])

    def test_bad_registration_or_missing_line_is_not_quality_evidence(self):
        ref=np.full((128,128),100.)
        fit=register_affine(ref,ref,(30,30,90,90))
        self.assertFalse(fit['accepted'])
        warp=np.array([[1.,0.,60.],[0.,1.,60.]])
        self.assertFalse(feature_profile(ref,ref,warp,(20,20,100,100),line_window=(45,75),y_range=(30,80))['valid'])

    def test_velocity_uses_timestamps_and_rejects_missing_frame_gap(self):
        tracks=[]
        for i in range(3):
            tracks.append({'index':i,'accepted':True,'correlation':.95,'sensor_ns':1_000_000_000+i*33_333_333,
                'exposure_us':4000,'warp':[[1,0,60+i*5],[0,1,60]],'determinant':1.,'angle_deg':0,
                'line':{'valid':True,'reference_line_slope':0.,'width_fwhm_px':3.,'depth_y':30.,'relative_contrast':.3}})
        rows=matched_velocity_rows(tracks)
        self.assertEqual(len(rows),1);self.assertAlmostEqual(rows[0]['speed_px_s'],150.,places=4)
        tracks[2]['sensor_ns']+=200_000_000
        self.assertEqual(matched_velocity_rows(tracks),[])


if __name__=='__main__':unittest.main()
