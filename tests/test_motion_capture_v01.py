from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import road_pi_motion_v01 as motion
import road_pi_controls_v01 as lab
from test_static_trial_v01 import TrialCamera,TrialRequest
from test_observer_v01 import Clock


class MotionCaptureTests(unittest.TestCase):
    def spec(self):
        return motion.MotionSpec(motion.TrialSpec('shutter-auto-gain',(0,0,8,8),(8,0,16,8),(2.,1.8),0.,4000),
                                 (80,80,144,144),duration_s=2,countdown_s=1)
    def run_fake(self,path,camera,**kwargs):
        with patch.object(lab,'system_metrics',return_value={}),patch.object(lab,'package_versions',return_value={}):
            return motion.run_capture(lambda:camera,path,self.spec(),clock=camera.clock,
                metrics=lambda:{'throttled':'throttled=0x0'},**kwargs)

    def test_countdown_frame_provenance_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            camera=TrialCamera(Clock());path=Path(tmp)/'new';phases=[]
            result=self.run_fake(path,camera,progress=lambda **s:phases.append(s['phase']))
            self.assertIsNone(result['error']);self.assertGreaterEqual(result['frames'],55)
            rows=[json.loads(l) for l in (path/'observations.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),len(list(path.glob('frame_*.npy'))))
            self.assertEqual(rows[0]['elapsed_s'],0)
            self.assertGreater(rows[-1]['actual_metadata']['SensorTimestamp'],rows[0]['actual_metadata']['SensorTimestamp'])
            self.assertLess(result['max_sensor_gap_ms'],34)
            self.assertLess(phases.index('countdown'),phases.index('recording'))
            self.assertEqual(camera.events[-2:],['stop','close'])

    def test_stopped_or_bad_array_closes_camera(self):
        for reason in ('stop','array'):
            with self.subTest(reason=reason),tempfile.TemporaryDirectory() as tmp:
                camera=TrialCamera(Clock())
                context=patch.object(TrialRequest,'make_array',side_effect=ValueError('bad Y')) if reason=='array' else patch.object(lab,'package_versions',return_value={})
                with context,self.assertRaises((InterruptedError,ValueError)):
                    self.run_fake(Path(tmp)/'new',camera,stopped=lambda:reason=='stop')
                self.assertEqual(camera.events[-2:],['stop','close'])

    def test_invalid_roi_and_duration(self):
        for roi in ((0,0,1920,1080),(-1,0,80,80),(0,0,16,16)):
            with self.assertRaises(ValueError):motion.MotionSpec(self.spec().trial,roi)

    def test_duplicate_clicks_enqueue_only_one_trial(self):
        state={'phase':'idle','completed':0,'total':3};jobs=queue.Queue(maxsize=1);lock=threading.Lock()
        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted=list(pool.map(lambda _:motion.enqueue_trial(state,jobs,lock),range(16)))
        self.assertEqual(sum(accepted),1);self.assertEqual(jobs.get_nowait(),0)
        self.assertFalse(motion.enqueue_trial(state,jobs,lock))


if __name__=='__main__':unittest.main()
