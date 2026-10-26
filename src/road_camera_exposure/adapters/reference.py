"""Capture a colour reference and record actual white balance, not road geometry."""
import copy
from io import BytesIO
import time

from ..imaging import RunBudget
from .picamera2_support import configure_lab,setup_record,system_metrics


def capture(config,output,*,factory=None,clock=time.monotonic_ns,metrics=system_metrics):
    if factory is None:
        from picamera2 import Picamera2
        factory=lambda:Picamera2(config['camera_index'])
    budget=RunBudget(output,max_bytes=10_000_000,reserve_bytes=config['reserve_bytes'])
    camera=None;error=None;cleanup=[];result=None
    try:
        camera=factory();requested,actual=configure_lab(camera)
        controls={'AeEnable':True,'ExposureTime':0,'AnalogueGain':0,
                  'ExposureTimeMode':0,'AnalogueGainMode':0,'AwbEnable':True,
                  'AfMode':0,'LensPosition':config['lens']}
        budget.write_json('camera.json',{**setup_record(camera,requested,actual),'reference_controls':controls})
        camera.set_controls(controls);camera.start();start=clock()
        while clock()-start<7_000_000_000:
            req=None
            try:
                req=camera.wait(camera.capture_request(wait=False,flush=False),timeout=2)
                md=req.get_metadata()
                if clock()-start>=5_000_000_000 and md.get('AeState')==2:
                    gains=md.get('ColourGains')
                    if not gains or len(gains)!=2:raise ValueError('white balance metadata unavailable')
                    out=BytesIO();req.save('main',out,format='jpeg')
                    if len(out.getvalue())>3_000_000:raise ValueError('reference JPEG exceeds 3 MB')
                    budget.write_new('reference.jpg',out.getvalue());budget.write_json('metadata.json',md)
                    result=copy.deepcopy(config);result['colour_gains']=[float(v) for v in gains]
                    result['reference_note']='Measured AWB only; verify focus, roi and tracking_rectangle before run.'
                    budget.write_json('config.json',result)
                    break
            finally:
                if req is not None:req.release()
        if result is None:raise TimeoutError('native exposure did not converge for reference')
    except BaseException as failure:error=type(failure).__name__+': '+str(failure);raise
    finally:
        if camera:
            for name in ('stop','close'):
                try:getattr(camera,name)()
                except BaseException as failure:cleanup.append(str(failure))
        budget.write_json('summary.json',{'error':error,'cleanup_errors':cleanup,'system_after':metrics()})
        if cleanup and error is None:raise RuntimeError('camera cleanup failed')
    return result
