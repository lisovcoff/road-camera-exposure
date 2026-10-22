"""Bounded stationary manual focus/exposure comparison; camera-free dry-run.

ROI is explicit in native image coordinates; no automatic target identification.
Scores are relative image gradients, NOT calibrated resolution or motion blur.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from road_observer_v01 import RunBudget, json_safe, visible_y_plane
from road_pi_controls_v01 import configure_lab, setup_record
from road_pi_observe_v01 import LimitedJpegBuffer, METADATA_FIELDS, system_metrics

POSITIONS = [0., .2, .35, .5, .65, .8, 1., 1.3, 1.6, 2., .5, .35, .65]


def patch_metrics(y, rectangle):
    if not isinstance(y,np.ndarray) or y.dtype!=np.uint8 or y.ndim!=2:
        raise ValueError("uint8 Y plane required")
    if len(rectangle)!=4 or any(type(v) is not int for v in rectangle):
        raise ValueError("integer pixel rectangle required")
    x0,y0,x1,y1=rectangle
    if not (0<=x0<x1<=y.shape[1] and 0<=y0<y1<=y.shape[0] and x1-x0>=8 and y1-y0>=8):
        raise ValueError("ROI outside image or too small")
    p=y[y0:y1,x0:x1].astype(np.float32)
    dx=np.diff(p,axis=1);dy=np.diff(p,axis=0)
    lap=p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*p[1:-1,1:-1]
    return {"mean":float(p.mean()),"std":float(p.std()),"p05":float(np.percentile(p,5)),
            "p95":float(np.percentile(p,95)),"laplacian_variance":float(lap.var()),
            "gradient_energy":float((dx*dx).mean()+(dy*dy).mean()),
            "gradient_abs_p90":float(np.percentile(np.concatenate((abs(dx).ravel(),abs(dy).ravel())),90)),
            "dark_pct":float((p<=20).mean()*100),"clipped_pct":float((p>=235).mean()*100)}


def get_request(camera):
    return camera.wait(camera.capture_request(wait=False,flush=True),timeout=2)


def run(camera_factory,output,roi,exposure,gain,positions,mode="focus"):
    patch_metrics(np.zeros((1080,1920),dtype=np.uint8),roi)
    if (not 100<=exposure<=33000 or not 1.13<=gain<=16 or not positions or len(positions)>15
            or any(not 0<=p<=2 for p in positions)):
        raise ValueError("bounded focus experiment parameters required")
    budget=RunBudget(output,max_bytes=30_000_000,reserve_bytes=1_000_000_000)
    camera=None;error=None;cleanup=[];count=0;start=time.monotonic_ns()
    sequence=[(p,exposure,gain) for p in positions] if mode=="focus" else [
        (positions[0],e,gain) for e in (2000,4000,8000,16000,exposure)]
    try:
        camera=camera_factory();requested,actual=configure_lab(camera)
        bounds=camera.camera_controls["LensPosition"]
        if not bounds[0]<=min(positions)<=max(positions)<=bounds[1]:raise ValueError("lens range unavailable")
        setup=setup_record(camera,requested,actual)
        setup.update({"purpose":"stationary_focus_or_brightness_comparison","mode":mode,"roi_main_xyxy":roi,
            "requested_sequence":sequence,"max_output_bytes":budget.max_bytes,
            "source_sha256_focus":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "warning":"paper ROI only; distances/target dimensions are not inferred; no motion test"})
        budget.write_json("setup.json",setup)
        camera.set_controls({"AeEnable":False,"AfMode":0,"ExposureTime":exposure,"AnalogueGain":gain,"LensPosition":positions[0]})
        camera.start()
        with budget.open_log() as log:
            for index,(lens,exp,ag) in enumerate(sequence):
                sent=time.monotonic_ns()
                controls={"AeEnable":False,"AfMode":0,"ExposureTime":exp,"AnalogueGain":ag,"LensPosition":lens}
                camera.set_controls(controls)
                # Metadata readback AND settling time, not merely sleep then assume applied.
                settled=False
                while time.monotonic_ns()-sent<2_500_000_000:
                    req=get_request(camera)
                    try:
                        md=req.get_metadata();now=time.monotonic_ns()
                        if (now-sent>=800_000_000 and md["SensorTimestamp"]-md["ExposureTime"]*1000>sent
                                and abs(md["LensPosition"]-lens)<=.025 and abs(md["ExposureTime"]-exp)<=82
                                and abs(md["AnalogueGain"]-ag)<=.31):settled=True
                    finally:req.release()
                    if settled:break
                if not settled:raise TimeoutError("focus/exposure readback did not settle")
                for sample in range(3):
                    if time.monotonic_ns()-start>65_000_000_000:raise TimeoutError("total focus budget exceeded")
                    req=get_request(camera)
                    try:
                        md=req.get_metadata()
                        if (abs(md["LensPosition"]-lens)>.025 or abs(md["ExposureTime"]-exp)>82
                                or abs(md["AnalogueGain"]-ag)>.31):raise ValueError("controls drifted during comparison")
                        y=visible_y_plane(req.make_array("main"),actual["main"])
                        metrics=patch_metrics(y,roi)
                        filename=None
                        if sample==1:
                            filename=f"step_{index:02d}.jpg"
                            with LimitedJpegBuffer(3_000_000) as buf:
                                req.save("main",buf,format="jpeg")
                                budget.write_new(filename,buf.getvalue())
                        record={"step":index,"sample":sample,"dispatch_started_ns":sent,
                            "observed_ns":time.monotonic_ns(),"requested_controls":controls,
                            "actual_metadata":{k:md.get(k) for k in METADATA_FIELDS},
                            "metrics":metrics,"snapshot":filename,"available_storage_bytes":budget.available_bytes()}
                        budget.append_record(log,record);count+=1
                    finally:req.release()
    except BaseException as failure:
        error=type(failure).__name__+": "+str(failure);raise
    finally:
        if camera is not None:
            for method in ("stop","close"):
                try:getattr(camera,method)()
                except BaseException as failure:cleanup.append(method+": "+str(failure))
        summary={"samples":count,"error":error,"cleanup_errors":cleanup,"system_after":system_metrics(),
                 "available_storage_bytes":budget.available_bytes(),"elapsed_ns":time.monotonic_ns()-start}
        budget.write_json("summary.json",summary)
        if cleanup and error is None:raise RuntimeError("camera cleanup failed")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-camera",action="store_true");p.add_argument("--output",type=Path)
    p.add_argument("--roi",nargs=4,type=int);p.add_argument("--exposure-us",type=int);p.add_argument("--gain",type=float)
    p.add_argument("--mode",choices=("focus","exposure"),default="focus")
    p.add_argument("--lens",type=float,default=.5)
    a=p.parse_args(argv)
    if not a.run_camera:
        print(json.dumps({"dry_run":True,"camera_started":False,"positions":POSITIONS,"budget_bytes":30_000_000}));return 0
    if any(v is None for v in (a.output,a.roi,a.exposure_us,a.gain)):p.error("NEW output, explicit ROI, exposure and gain required")
    from picamera2 import Picamera2
    print(json.dumps(json_safe(run(lambda:Picamera2(0),a.output,a.roi,a.exposure_us,a.gain,
                                  POSITIONS if a.mode=="focus" else [a.lens],a.mode))))
    return 0


if __name__=="__main__":sys.exit(main())
