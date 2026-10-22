"""Stationary focus metrics and delayed readback lifecycle, no camera."""
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import road_pi_focus_v01 as focus
import road_pi_controls_v01 as manual
from test_controls_v01 import ManualCamera,ManualRequest
from test_observer_v01 import Clock

class FocusRequest(ManualRequest):
    def get_metadata(self):
        md=super().get_metadata();md['LensPosition']=self.camera.lens;return md
    def make_array(self,stream):
        assert stream=='main'
        return np.full((1620,1920),100,dtype=np.uint8)

class FocusCamera(ManualCamera):
    camera_controls={**ManualCamera.camera_controls,'LensPosition':(0.,15.,1.)}
    def __init__(self,clock):super().__init__(clock);self.lens=1.
    def capture_request(self,**kwargs):
        assert kwargs=={'wait':False,'flush':True};return object()
    def wait(self,job,timeout):
        target=self.queued[1]['LensPosition'] if self.queued else self.lens
        super().wait(job,timeout)
        if self.queued is None:self.lens=target
        return FocusRequest(self)

class FocusTests(unittest.TestCase):
    def test_edge_metrics_and_bounds(self):
        flat=np.full((40,40),100,dtype=np.uint8);edge=flat.copy();edge[:,20:]=200
        self.assertEqual(focus.patch_metrics(flat,[0,0,40,40])['gradient_energy'],0)
        self.assertGreater(focus.patch_metrics(edge,[0,0,40,40])['gradient_energy'],0)
        for rect in ([-1,0,20,20],[0,0,100,20],[0,0,2,2]):
            with self.assertRaises(ValueError):focus.patch_metrics(flat,rect)
    def test_wait_for_lens_and_cleanup(self):
        camera=FocusCamera(Clock())
        with tempfile.TemporaryDirectory() as d,patch.object(focus.time,'monotonic_ns',camera.clock), \
             patch.object(focus,'system_metrics',return_value={}), \
             patch.object(manual,'system_metrics',return_value={}), \
             patch.object(manual,'package_versions',return_value={}):
            result=focus.run(lambda:camera,Path(d)/'new',[800,440,1020,550],4000,8,[.2,.5])
            self.assertEqual(result['samples'],6);self.assertIsNone(result['error'])
            self.assertEqual(camera.events[-2:],['stop','close'])
            self.assertEqual(len(list((Path(d)/'new').glob('*.jpg'))),2)
    def test_failure_releases_request_and_camera(self):
        camera=FocusCamera(Clock())
        with tempfile.TemporaryDirectory() as d,patch.object(focus.time,'monotonic_ns',camera.clock), \
             patch.object(focus,'system_metrics',return_value={}), \
             patch.object(manual,'system_metrics',return_value={}), \
             patch.object(manual,'package_versions',return_value={}), \
             patch.object(FocusRequest,'make_array',side_effect=ValueError('bad Y')):
            with self.assertRaisesRegex(ValueError,'bad Y'):
                focus.run(lambda:camera,Path(d)/'new',[800,440,1020,550],4000,8,[.5])
            self.assertIn('release',camera.events);self.assertEqual(camera.events[-2:],['stop','close'])

if __name__=='__main__':unittest.main()
