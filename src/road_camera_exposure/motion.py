"""Laboratory target-motion shutter policy, with native gain. No hardware I/O.

This tracker is tied to an explicit target rectangle, NOT a road-wide speed
estimator. Unknown tracking selects a conservative 4ms demonstration fallback.
"""
import math
import numpy as np
from .motion_analysis import register_affine


class TargetTracker:
    def __init__(self,rectangle=(355,45,555,210),factor=4):
        self.rectangle=tuple(int(v//factor) for v in rectangle);self.factor=factor
        self.reference=None;self.warp=None;self.previous=None;self.previous_ns=None

    def observe(self,crop,midpoint_ns):
        if not isinstance(crop,np.ndarray) or crop.ndim!=2 or crop.dtype!=np.uint8:
            raise ValueError('uint8 target crop required')
        f=self.factor;h,w=crop.shape
        small=crop[:h//f*f,:w//f*f].reshape(h//f,f,w//f,f).mean(axis=(1,3))
        if self.reference is None:self.reference=small.copy()
        fit=register_affine(self.reference,small,self.rectangle,initial=self.warp)
        valid=fit['accepted'] and fit['correlation']>=.8
        speed=None
        if valid:
            self.warp=fit['warp']
            if self.previous is not None and 0<midpoint_ns-self.previous_ns<=350_000_000:
                points=np.array([[-10,0,1],[-20,-15,1],[20,-15,1],[20,15,1],[-20,15,1]]).T
                displacement=(self.warp-self.previous)@points
                speed=float(np.max(np.hypot(displacement[0],displacement[1]))*f/((midpoint_ns-self.previous_ns)/1e9))
            self.previous=self.warp.copy();self.previous_ns=midpoint_ns
        else:self.previous=None;self.previous_ns=None
        return {'speed_px_s':speed,'valid':valid and speed is not None,
                'correlation':fit['correlation'],'angle_deg':fit['angle_deg']}


class ShutterPolicy:
    def __init__(self):
        self.still_since=None;self.mode='UNKNOWN';self.last_exposure=4000

    def update(self,speed,valid,now_ns):
        if type(now_ns) is not int or now_ns<0:raise ValueError('monotonic ns required')
        if valid and (type(speed) not in (int,float) or not math.isfinite(speed) or speed<0):
            raise ValueError('finite nonnegative speed required')
        if not valid:
            self.still_since=None;self.mode='UNKNOWN';self.last_exposure=4000
            return {'mode':self.mode,'exposure_us':4000,'reason':'tracking unknown; lab fallback, no blur guarantee'}
        if speed>=30:
            self.mode='MOVING';self.still_since=None
        elif speed<10:
            if self.still_since is None:self.still_since=now_ns
            if now_ns-self.still_since>=1_000_000_000:self.mode='STILL'
        else:self.still_since=None
        if self.mode=='STILL':
            return {'mode':'STILL','exposure_us':None,'reason':'native shutter and gain after 1s stillness'}
        cap=int(.8e6/(speed*1.2+25))-42
        exposure=max(1000,min(8000,cap))
        if abs(exposure-self.last_exposure)/max(self.last_exposure,1)>=.15:
            self.last_exposure=exposure
        return {'mode':self.mode,'exposure_us':self.last_exposure,
                'reason':'0.8px heuristic budget with motion margin' if cap>=1000 else 'minimum exposure reached; blur budget infeasible'}


class ControlGate:
    def __init__(self):
        self.pending=None;self.last=None;self.mode_cache=None;self.mode_observed_ns=None
        self.sent_count=0;self.observed_count=0

    def update(self,desired,metadata,now_ns):
        if metadata.get('ExposureTimeMode') is not None:
            self.mode_cache=metadata['ExposureTimeMode'];self.mode_observed_ns=now_ns
        start=metadata['SensorTimestamp']-metadata['ExposureTime']*1000
        if self.pending is not None:
            exp,sent=self.pending
            if now_ns-sent>2_000_000_000:raise TimeoutError('hybrid exposure application timeout')
            manual=exp is not None
            matched=(abs(metadata['ExposureTime']-exp)<=42 if manual else
                     self.mode_cache==0 and self.mode_observed_ns is not None and self.mode_observed_ns>sent)
            if start>sent and matched:
                self.pending=None;self.last=exp;self.observed_count+=1
            else:return None
        exp=desired['exposure_us']
        if exp==self.last:return None
        return {'AeEnable':True,'ExposureTime':0 if exp is None else exp,
                'ExposureTimeMode':0 if exp is None else 1,'AnalogueGain':0,'AnalogueGainMode':0}

    def sent(self,controls,now_ns):
        if self.pending is not None:raise RuntimeError('only one command in flight')
        self.pending=(controls['ExposureTime'] or None,now_ns);self.sent_count+=1
