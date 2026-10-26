import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'tools'))
import road_pi_trial_v01 as trial
import road_pi_controls_v01 as lab
from analyze_static_trial_v01 import analyze
from test_observer_v01 import Clock,FakeCamera,FakeRequest


class TrialRequest(FakeRequest):
    def get_metadata(self):
        md=super().get_metadata();c=self.camera;v=c.applied
        md.update(ExposureTime=(v['ExposureTime']-19 if v['ExposureTime'] else 8000),
            AnalogueGain=4.,ExposureTimeMode=v['ExposureTimeMode'],AnalogueGainMode=v['AnalogueGainMode'],
            LensPosition=v['LensPosition'],ColourGains=v['ColourGains'],AeState=2)
        return md
    def make_array(self,stream):
        assert stream=='main'
        a=np.full((1620,1920),80,dtype=np.uint8)
        a[100:116,100:116]=140
        return a


class TrialCamera(FakeCamera):
    camera_controls={**FakeCamera.camera_controls,
        **{k:(0,10,0) for k in trial.TrialSpec('native-normal',(100,100,116,116),
            (130,100,146,116),(2.,1.8),0.).controls()},
        'ExposureTime':(13,112015096,20000),'AnalogueGain':(1.1228,16.,1.),'LensPosition':(0,15,1)}
    def __init__(self,clock):
        super().__init__(clock);self.sent=[];self.pending=None;self.applied={}
    def create_video_configuration(self,**kw):return kw
    def set_controls(self,v):self.sent.append(v);self.pending=(3,v)
    def capture_request(self,**kw):
        assert kw=={'wait':False,'flush':False};return object()
    def wait(self,job,timeout):
        self.clock.now+=33_333_333
        n,v=self.pending
        if n:self.pending=(n-1,v)
        else:self.applied=v
        # Before application return explicit manual-mode mismatch, not a fake ACK.
        if not self.applied:
            self.applied={**v,'ExposureTime':2000,'ExposureTimeMode':1,'AnalogueGainMode':1}
        return TrialRequest(self)


class TrialTests(unittest.TestCase):
    def spec(self,mode='native-normal'):
        return trial.TrialSpec(mode,(100,100,116,116),(130,100,146,116),(2.,1.8),0.,4000,5)
    def run_fake(self,path,camera,spec=None):
        with patch.object(lab,'system_metrics',return_value={'throttled':'throttled=0x0'}), \
             patch.object(lab,'package_versions',return_value={'fake':True}):
            return trial.run(lambda:camera,path,spec or self.spec(),clock=camera.clock,
                             metrics=lambda:{'throttled':'throttled=0x0'})

    def test_hybrid_sets_manual_shutter_and_auto_gain(self):
        c=self.spec('shutter-auto-gain').controls()
        self.assertEqual((c['ExposureTime'],c['ExposureTimeMode'],c['AnalogueGain'],c['AnalogueGainMode']),(4000,1,0,0))
        self.assertFalse(c['AwbEnable'])
        self.assertEqual(self.spec('native-short').controls()['AeExposureMode'],1)

    def test_invalid_overlapping_or_large_rois_rejected(self):
        for roi in ((100,100,116,116),(0,0,80,80),(-1,0,16,16)):
            with self.assertRaises(ValueError):
                trial.TrialSpec('native-normal',(100,100,116,116),roi,(2.,1.8),0.)

    def test_actual_acquisition_and_analysis_lifecycle(self):
        for mode in ('native-normal','native-short','shutter-auto-gain'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                camera=TrialCamera(Clock());path=Path(tmp)/'new'
                result=self.run_fake(path,camera,self.spec(mode))
                self.assertIsNone(result['error']);self.assertGreaterEqual(result['patch_pairs'],4)
                self.assertEqual(camera.events[-2:],['stop','close'])
                self.assertEqual(len(camera.sent),1)
                report=analyze(path)
                self.assertFalse(report['sensor_noise_measured']);self.assertIsNone(report['quality_ranking'])
                self.assertEqual(report['contrast_delta_y'],60)
                self.assertIsNone(report['contrast_variation_proxy']) # Zero variance is not infinite quality.

    def test_bad_mode_or_patch_failure_releases_camera(self):
        for failure in ('wrong-mode','array-error'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                camera=TrialCamera(Clock())
                original=TrialRequest.get_metadata
                def wrong(req):
                    md=original(req);md['AnalogueGainMode']=1;return md
                context=patch.object(TrialRequest,'get_metadata',wrong) if failure=='wrong-mode' else \
                    patch.object(TrialRequest,'make_array',side_effect=ValueError('bad array'))
                with context,self.assertRaises((TimeoutError,ValueError)):
                    self.run_fake(Path(tmp)/'new',camera)
                self.assertIn('release',camera.events);self.assertEqual(camera.events[-2:],['stop','close'])

    def test_wrong_awb_or_missing_ae_cannot_confirm_readback(self):
        md={'ExposureTime':4000,'ExposureTimeMode':0,'AnalogueGainMode':0,'AeState':2,
            'LensPosition':0.,'ColourGains':[2.,1.8]}
        self.assertTrue(trial.check_readback(md,self.spec())[0])
        for update in ({'ColourGains':[1.,1.]},{'AeState':None},{'ExposureTimeMode':1}):
            self.assertFalse(trial.check_readback({**md,**update},self.spec())[0])

    def test_failed_summary_and_path_escape_rejected_by_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            camera=TrialCamera(Clock());path=Path(tmp)/'new';self.run_fake(path,camera)
            log=path/'observations.jsonl';rows=[json.loads(l) for l in log.read_text().splitlines()]
            rows[0]['patches']['target']='../outside.npy'
            log.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError,'unsafe patch'):
                analyze(path)

    def test_temporal_variation_and_drift_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            camera=TrialCamera(Clock());path=Path(tmp)/'new';self.run_fake(path,camera)
            rows=[json.loads(l) for l in (path/'observations.jsonl').read_text().splitlines()]
            rng=np.random.default_rng(19)
            patches=[r for r in rows if r['patches']]
            for row in patches:
                for name,base in [('target',140),('background',80)]:
                    array=np.rint(base+rng.normal(0,3,(16,16))).clip(0,255).astype(np.uint8)
                    np.save(path/row['patches'][name],array,allow_pickle=False)
            report=analyze(path)
            self.assertTrue(report['stationarity_checks_passed'])
            self.assertAlmostEqual(report['patches']['target']['temporal_difference_std_y'],3,delta=.4)
            self.assertIsNotNone(report['contrast_variation_proxy'])
            np.save(path/patches[-1]['patches']['target'],np.full((16,16),180,dtype=np.uint8),allow_pickle=False)
            report=analyze(path)
            self.assertFalse(report['stationarity_checks_passed'])
            self.assertIsNone(report['contrast_variation_proxy'])

    def test_unsupported_control_and_power_fault_stop_before_results(self):
        for failure in ('unsupported','power'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                camera=TrialCamera(Clock())
                if failure=='unsupported':
                    camera.camera_controls={k:v for k,v in camera.camera_controls.items() if k!='AnalogueGainMode'}
                with patch.object(lab,'system_metrics',return_value={}),patch.object(lab,'package_versions',return_value={}):
                    with self.assertRaises((ValueError,RuntimeError)):
                        trial.run(lambda:camera,Path(tmp)/'new',self.spec(),clock=camera.clock,
                                  metrics=lambda:{'throttled':'throttled=0x50005'})
                self.assertEqual(camera.events[-2:],['stop','close'])


if __name__=='__main__':unittest.main()
