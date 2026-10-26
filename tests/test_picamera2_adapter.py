from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from road_camera_exposure.config import load,runtime_args
from road_camera_exposure.adapters import picamera2 as app
from road_camera_exposure.adapters import picamera2_support as support
from road_camera_exposure.adapters.reference import capture

ROOT=Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):self.now=10_000_000_000
    def ns(self):return self.now
    def seconds(self):return self.now/1e9


class Request:
    def __init__(self,c):self.c=c
    def get_metadata(self):
        v=self.c.applied;exposure=v.get('ExposureTime',0) or 8000
        return {'SensorTimestamp':self.c.clock.now-1_000_000,'ExposureTime':max(13,exposure-19),
            'AnalogueGain':4.,'DigitalGain':1.,'FrameDuration':33333,'AeState':2,
            'LensPosition':v.get('LensPosition',0.),'ColourGains':v.get('ColourGains',[1.8,2.]),
            'ExposureTimeMode':v.get('ExposureTimeMode',0),'AnalogueGainMode':0}
    def make_array(self,name):return self.c.array
    def save(self,name,output,format):output.write(b'FAKE-JPEG-FOR-LIFECYCLE-TEST-ONLY')
    def release(self):self.c.events.append('release')


class Camera:
    camera_properties={'Model':'imx708_FAKE'}
    camera_controls={k:(0,16,0) for k in support.TrialSpec('native-normal',(0,0,8,8),(8,0,16,8),(1.8,2.),0.).controls()}
    camera_controls.update(ExposureTime=(13,112015096,20000),AnalogueGain=(1.1228,16.,1.),LensPosition=(0,15,1))
    def __init__(self):
        self.clock=Clock();self.events=[];self.applied={};self.pending=None
        self.array=np.full((1620,1920),128,dtype=np.uint8)
        self.array[360:720,640:1280]=np.random.default_rng(2).integers(30,200,(360,640),dtype=np.uint8)
    def create_video_configuration(self,**kw):return kw
    def configure(self,c):pass
    def camera_configuration(self):
        return {'main':{'size':(1920,1080),'format':'YUV420','stride':1920},
            'lores':{'size':(320,180),'format':'YUV420','stride':384},
            'sensor':{'output_size':(2304,1296),'bit_depth':10},'transform':'identity'}
    def set_controls(self,c):self.pending=(2,c)
    def start(self):self.events.append('start')
    def stop(self):self.events.append('stop')
    def close(self):self.events.append('close')
    def capture_request(self,**kw):return object()
    def wait(self,job,timeout):
        self.clock.now+=33_333_333
        if self.pending:
            n,v=self.pending
            if n:self.pending=(n-1,v)
            else:self.applied={**self.applied,**v};self.pending=None
        return Request(self)


class AdapterTests(unittest.TestCase):
    def config(self):
        c=load(ROOT/'configs/module3.example.json');c['colour_gains']=[1.8,2.];c['duration_s']=10;return c
    def test_reference_saves_configuration_and_releases_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            camera=Camera()
            with patch.object(support,'system_metrics',return_value={}),patch.object(support,'package_versions',return_value={}):
                result=capture(self.config(),Path(tmp)/'ref',factory=lambda:camera,clock=camera.clock.ns,metrics=lambda:{})
            self.assertEqual(result['colour_gains'],[1.8,2.]);self.assertEqual(camera.events[-2:],['stop','close'])
            self.assertTrue((Path(tmp)/'ref/config.json').exists())
    def test_runner_mock_camera_colour_preview_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            import socket
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            c=self.config();c['port']=port;camera=Camera();args=runtime_args(c,Path(tmp)/'run')
            fake_clock=SimpleNamespace(monotonic=camera.clock.seconds,monotonic_ns=camera.clock.ns)
            sdk=SimpleNamespace(Picamera2=lambda _:camera)
            jpeg=SimpleNamespace(encode_jpeg_yuv_planes=lambda *args:b'\xff\xd8MOCK\xff\xd9')
            with patch.dict(sys.modules,{'picamera2':sdk,'simplejpeg':jpeg}), \
                 patch.object(app,'time',fake_clock),patch.object(app.signal,'signal'), \
                 patch.object(app,'system_metrics',return_value={'throttled':'throttled=0x0'}), \
                 patch.object(support,'system_metrics',return_value={}),patch.object(support,'package_versions',return_value={}):
                app.run(args)
            summary=json.loads((Path(tmp)/'run/summary.json').read_text())
            self.assertIsNone(summary['error']);self.assertGreater(summary['samples'],200)
            self.assertEqual(camera.events[-2:],['stop','close'])
    def test_reference_capture_failure_releases_request_and_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            camera=Camera()
            with patch.object(support,'system_metrics',return_value={}),patch.object(support,'package_versions',return_value={}), \
                 patch.object(Request,'save',side_effect=ValueError('capture failed')):
                with self.assertRaisesRegex(ValueError,'capture failed'):
                    capture(self.config(),Path(tmp)/'ref',factory=lambda:camera,clock=camera.clock.ns,metrics=lambda:{})
            self.assertEqual(camera.events[-2:],['stop','close']);self.assertIn('release',camera.events)


if __name__=='__main__':unittest.main()
