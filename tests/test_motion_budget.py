from pathlib import Path
import sys
import unittest

import numpy as np

from road_camera_exposure.motion_budget import projected_velocity, exposure_budget, diagnostic_target_cnr


class MotionTests(unittest.TestCase):
    def budget(self, flow, **updates):
        kwargs = dict(blur_limit_px=1., uncertainty_px_s=0., now_ns=100,
                      measured_ns=90, max_age_ns=20, frame_duration_us=33333,
                      min_exposure_us=100, sensor_max_us=32680,
                      readback_tolerance_us=42, geometry_calibrated=True)
        kwargs.update(updates)
        return exposure_budget(flow, **kwargs)

    def test_stationary_points_and_lateral_translation(self):
        self.assertTrue(np.allclose(projected_velocity([[0,0,2]], [0,0,0], [0,0,0],1000,1000),0))
        v=projected_velocity([[0,0,2],[0,0,4]], [1,0,0], [0,0,0],1000,1000)
        np.testing.assert_allclose(v,[[-500,0],[-250,0]])

    def test_forward_motion_and_yaw_signs(self):
        np.testing.assert_allclose(projected_velocity([[1,0,2]], [0,0,1],[0,0,0],1000,1000),[[250,0]])
        np.testing.assert_allclose(projected_velocity([[0,0,2]], [0,0,0],[0,1,0],1000,1000),[[-1000,0]])

    def test_projection_matches_finite_difference(self):
        p=np.array([[.5,.8,3.],[-1.,.3,2.]])
        v=np.array([.1,.2,1.]); w=np.array([.02,-.03,.01]);dt=1e-7
        p2=p+(-v-np.cross(w,p))*dt
        projected=lambda q:q[:,:2]/q[:,2,None]*[800,900]
        np.testing.assert_allclose(projected_velocity(p,v,w,800,900),(projected(p2)-projected(p))/dt,rtol=1e-6)

    def test_invalid_geometry(self):
        for points in ([[0,0,0]],[[0,0,-1]],[[float('nan'),0,1]],[]):
            with self.assertRaises(ValueError):
                projected_velocity(points,[0,0,0],[0,0,0],1000,1000)

    def test_actual_tolerance_is_inside_blur_budget(self):
        b=self.budget([[0,1000]])
        self.assertEqual(b.max_requested_exposure_us,958)
        self.assertLessEqual((b.max_requested_exposure_us+42)*1000/1e6,1.)
        self.assertFalse(b.dispatch_permitted)

    def test_worst_point_uncertainty_and_frame_cap(self):
        self.assertEqual(self.budget([[0,10],[0,800]],uncertainty_px_s=200.).max_requested_exposure_us,958)
        self.assertEqual(self.budget([[0,0]]).max_actual_exposure_us,32680)
        self.assertEqual(self.budget([[0,0]],frame_duration_us=10000).max_actual_exposure_us,10000)

    def test_stale_future_and_infeasible_motion(self):
        for measured in (0,101):
            self.assertIsNone(self.budget([[1,0]],measured_ns=measured).max_requested_exposure_us)
        self.assertEqual(self.budget([[1e7,0]]).status,'BLUR_CONSTRAINT_INFEASIBLE')
        self.assertEqual(self.budget([[0,0]],geometry_calibrated=False).status,'UNCALIBRATED_MODEL_ONLY')

    def test_model_has_low_light_and_static_counterexamples(self):
        def score(t,light=100.,motion=0.):
            return diagnostic_target_cnr(electrons_per_ms=light,exposure_us=t,read_noise_e=3.,
                    target_contrast=.5,target_width_px=4.,velocity_px_s=motion)
        self.assertGreater(score(32000),score(2000))  # Static: short shutter loses SNR.
        self.assertGreater(score(2000,motion=2000.),score(32000,motion=2000.))
        self.assertLess(score(2000,light=.01,motion=2000.),.01)  # Fast but unusably dark.


if __name__ == '__main__':
    unittest.main()
