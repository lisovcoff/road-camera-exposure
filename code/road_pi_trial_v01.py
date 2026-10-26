"""Bounded static trial: native Normal/Short or fixed shutter + native gain.

No hardware import by default. No network, encoder, installation or firmware
changes. Small uncompressed post-ISP Y patches are not sensor RAW data.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from road_observer_v01 import RunBudget, frame_from_metadata, json_safe, visible_y_plane
from road_pi_controls_v01 import configure_lab, setup_record
from road_pi_observe_v01 import LimitedJpegBuffer, METADATA_FIELDS, system_metrics


def validate_roi(roi):
    if not isinstance(roi, (tuple, list)) or len(roi) != 4 or any(type(x) is not int for x in roi):
        raise ValueError('ROI must be integer x0 y0 x1 y1')
    x0,y0,x1,y1=roi
    if not (0<=x0<x1<=1920 and 0<=y0<y1<=1080 and 8<=x1-x0<=64 and 8<=y1-y0<=64):
        raise ValueError('main-image patch must be 8..64 pixels per side')


@dataclass(frozen=True)
class TrialSpec:
    mode: str
    target_roi: tuple
    background_roi: tuple
    colour_gains: tuple
    lens: float
    exposure_us: int = 4000
    duration_s: int = 10

    def __post_init__(self):
        if self.mode not in ('native-normal','native-short','shutter-auto-gain'):
            raise ValueError('unsupported trial mode')
        for roi in (self.target_roi,self.background_roi):
            validate_roi(roi)
        a,b=self.target_roi,self.background_roi
        if max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3]):
            raise ValueError('target/background patches must not overlap')
        if (not isinstance(self.colour_gains,(tuple,list)) or len(self.colour_gains)!=2
                or any(type(v) not in (int,float) or not math.isfinite(v) or not .1<=v<=8 for v in self.colour_gains)):
            raise ValueError('two measured colour gains required in .1..8')
        if type(self.lens) not in (float,int) or not math.isfinite(self.lens) or not 0<=self.lens<=2:
            raise ValueError('lens candidate must be in 0..2')
        if type(self.exposure_us) is not int or not 100<=self.exposure_us<=12000:
            raise ValueError('fixed shutter experiment limited to 100..12000 us')
        if type(self.duration_s) is not int or not 5<=self.duration_s<=30:
            raise ValueError('duration must be 5..30 seconds')

    def controls(self):
        manual=self.mode=='shutter-auto-gain'
        return {'AeEnable':True, 'ExposureTime':self.exposure_us if manual else 0,
                'AnalogueGain':0, 'ExposureTimeMode':1 if manual else 0, 'AnalogueGainMode':0,
                'AeExposureMode':1 if self.mode=='native-short' else 0,
                'AeMeteringMode':0, 'AeConstraintMode':0, 'ExposureValue':0.,
                'AfMode':0, 'LensPosition':self.lens,
                'AwbEnable':False, 'ColourGains':tuple(self.colour_gains)}


def check_readback(md,spec):
    """Mode fields may be emitted only when changed; absence is explicit."""
    reported = md.get('ExposureTimeMode'),md.get('AnalogueGainMode')
    wanted = (1 if spec.mode=='shutter-auto-gain' else 0,0)
    if any(a is not None and a!=b for a,b in zip(reported,wanted)):
        return False,'wrong_control_mode'
    if md.get('AeState') not in (1,2):
        return False,'native_AE_not_active_or_unreported'
    if (spec.mode=='shutter-auto-gain' and abs(md['ExposureTime']-spec.exposure_us)>42):
        return False,'fixed_shutter_not_observed'
    if md.get('LensPosition') is None or abs(md['LensPosition']-spec.lens)>.025:
        return False,'focus_not_observed'
    gains=md.get('ColourGains')
    if gains is None or len(gains)!=2 or any(abs(a-b)>.05 for a,b in zip(gains,spec.colour_gains)):
        return False,'fixed_AWB_not_observed'
    return True,'reported_modes' if None not in reported else 'AE_active_modes_partly_unreported'


def run(camera_factory,output,spec,*,clock=time.monotonic_ns,metrics=system_metrics):
    if not isinstance(spec,TrialSpec):
        raise ValueError('validated TrialSpec required')
    budget=RunBudget(output,max_bytes=20_000_000,reserve_bytes=1_000_000_000)
    camera=None;error=None;cleanup=[];count=0;patch_count=0
    started=clock();first=None;last_frame=None;settled=0;next_sample=next_patch=next_metrics=0
    snapshots=set();readback_labels=set()
    try:
        camera=camera_factory();requested,actual=configure_lab(camera)
        wanted=spec.controls()
        unsupported=set(wanted)-set(camera.camera_controls)
        if unsupported:
            raise ValueError('required camera controls unavailable: '+','.join(sorted(unsupported)))
        controls_range=camera.camera_controls
        if not controls_range['LensPosition'][0]<=spec.lens<=controls_range['LensPosition'][1]:
            raise ValueError('lens outside actual camera bounds')
        setup=setup_record(camera,requested,actual)
        setup.update({'schema_version':1,'trial':asdict(spec),'requested_controls':wanted,
            'focus':'explicit candidate; not road calibrated','AWB':'fixed supplied reference',
            'purpose':'static response and post-ISP temporal variation; not motion/detection proof',
            'source_trial_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'max_output_bytes':budget.max_bytes,'max_loop_s':spec.duration_s+7})
        budget.write_json('setup.json',setup)
        camera.set_controls(wanted);sent=clock();camera.start()
        with budget.open_log() as log:
            while count<301:
                if clock()-started>(spec.duration_s+7)*1_000_000_000:
                    raise TimeoutError('total experiment deadline')
                req=None
                try:
                    req=camera.wait(camera.capture_request(wait=False,flush=False),timeout=2)
                    md=req.get_metadata();now=clock();frame=frame_from_metadata(md)
                    if not 0<=now-frame.exposure_start_ns<=500_000_000 or (
                            last_frame is not None and frame.frame_id<=last_frame):
                        raise ValueError('stale, duplicate or incompatible frame clock')
                    last_frame=frame.frame_id
                    ag=camera.camera_controls['AnalogueGain']
                    if not ag[0]-.02<=frame.analogue_gain<=ag[1]+.02 or frame.exposure_us>frame.frame_duration_us:
                        raise ValueError('invalid exposure/gain readback')
                    ok,label=check_readback(md,spec)
                    if first is None:
                        # Three fresh matching frames and 2 seconds warmup; AE must
                        # report converged, not merely any gain value.
                        settled=settled+1 if ok and md['AeState']==2 and frame.exposure_start_ns>sent else 0
                        if now-sent>=2_000_000_000 and settled>=3:
                            first=now
                        elif now-sent>5_000_000_000:
                            raise TimeoutError('mode/focus/AWB/AE readback did not settle: '+label)
                        else:
                            continue
                    if not ok:
                        raise ValueError('controls drifted: '+label)
                    elapsed=(now-first)/1e9
                    if elapsed>=spec.duration_s:
                        break
                    if elapsed<next_sample:
                        continue
                    next_sample=elapsed+.1
                    readback_labels.add(label)
                    if elapsed>=next_metrics:
                        health=metrics()
                        if health.get('throttled')!='throttled=0x0':
                            raise RuntimeError('power status is not verified clean')
                        next_metrics=elapsed+2
                    plane=visible_y_plane(req.make_array('main'),actual['main'])
                    means={};patch_files={}
                    for name,roi in (('target',spec.target_roi),('background',spec.background_roi)):
                        x0,y0,x1,y1=roi
                        patch=plane[y0:y1,x0:x1].copy()
                        means[name]={'mean_y':float(patch.mean()),'dark_pct':float((patch<=20).mean()*100),
                                     'clipped_pct':float((patch>=235).mean()*100)}
                        if elapsed>=next_patch:
                            filename=f'{name}_{patch_count:03d}.npy'
                            buf=BytesIO();np.save(buf,patch,allow_pickle=False)
                            budget.write_new(filename,buf.getvalue());patch_files[name]=filename
                    if patch_files:
                        patch_count+=1;next_patch=elapsed+1
                    snap=None
                    stage=0 if elapsed<spec.duration_s/2 else (1 if elapsed<spec.duration_s-1 else 2)
                    if stage not in snapshots:
                        snap=f'scene_{stage:02d}.jpg'
                        with LimitedJpegBuffer(3_000_000) as buf:
                            req.save('main',buf,format='jpeg');budget.write_new(snap,buf.getvalue())
                        snapshots.add(stage)
                    budget.append_record(log,{'schema_version':1,'sample':count,'elapsed_s':elapsed,
                        'observed_ns':now,'dispatch_started_ns':sent,'requested_controls':wanted,
                        'actual_metadata':{k:md.get(k) for k in METADATA_FIELDS},
                        'readback':label,'patches':patch_files,'patch_stats':means,'snapshot':snap,
                        'system':health,'available_storage_bytes':budget.available_bytes()})
                    count+=1
                finally:
                    if req is not None:req.release()
    except BaseException as failure:
        error=type(failure).__name__+': '+str(failure);raise
    finally:
        if camera is not None:
            for name in ('stop','close'):
                try:getattr(camera,name)()
                except BaseException as failure:cleanup.append(name+': '+str(failure))
        result={'samples':count,'patch_pairs':patch_count,'error':error,'cleanup_errors':cleanup,
                'readback_labels':sorted(readback_labels),'system_after':metrics(),
                'available_storage_bytes':budget.available_bytes(),'elapsed_ns':clock()-started}
        budget.write_json('summary.json',result)
        if cleanup and error is None:raise RuntimeError('cleanup failed')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-camera',action='store_true');p.add_argument('--allow-uncalibrated-lab',action='store_true')
    p.add_argument('--mode',choices=('native-normal','native-short','shutter-auto-gain'),default='native-normal')
    p.add_argument('--output',type=Path);p.add_argument('--target-roi',nargs=4,type=int)
    p.add_argument('--background-roi',nargs=4,type=int);p.add_argument('--colour-gains',nargs=2,type=float)
    p.add_argument('--lens',type=float);p.add_argument('--exposure-us',type=int,default=4000)
    p.add_argument('--duration',type=int,default=10)
    a=p.parse_args(argv)
    if not a.run_camera:
        print(json.dumps({'dry_run':True,'camera_started':False,'mode':a.mode,
                         'max_bytes':20_000_000,'snapshots':3,'network':False}));return 0
    if not a.allow_uncalibrated_lab or any(v is None for v in (a.output,a.target_roi,a.background_roi,a.colour_gains,a.lens)):
        p.error('explicit lab opt-in, new output, both ROIs, measured colour gains and lens required')
    spec=TrialSpec(a.mode,tuple(a.target_roi),tuple(a.background_roi),tuple(a.colour_gains),a.lens,a.exposure_us,a.duration)
    from picamera2 import Picamera2
    print(json.dumps(json_safe(run(lambda:Picamera2(0),a.output,spec))))
    return 0


if __name__=='__main__':sys.exit(main())
