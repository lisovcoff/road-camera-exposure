"""Bounded, user-triggered motion feasibility capture; default camera-free dry run.

Uncompressed post-ISP main Y crop + actual sensor timestamps, no video codec.
The local page starts one trial at a time; camera is closed while waiting.
"""
import argparse
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
from io import BytesIO
import ipaddress
import json
from pathlib import Path
import queue
import secrets
import signal
import threading
import time

import numpy as np

from road_observer_v01 import RunBudget, frame_from_metadata, json_safe, visible_y_plane
from road_pi_controls_v01 import configure_lab, setup_record
from road_pi_observe_v01 import METADATA_FIELDS, system_metrics
from road_pi_trial_v01 import TrialSpec, check_readback


@dataclass(frozen=True)
class MotionSpec:
    trial: TrialSpec
    roi: tuple
    duration_s: int = 10
    countdown_s: int = 5

    def __post_init__(self):
        if not isinstance(self.trial,TrialSpec):raise ValueError('validated trial required')
        r=self.roi
        if (not isinstance(r,(tuple,list)) or len(r)!=4 or any(type(x) is not int for x in r)
                or not (0<=r[0]<r[2]<=1920 and 0<=r[1]<r[3]<=1080
                        and 32<=r[2]-r[0]<=640 and 32<=r[3]-r[1]<=360)):
            raise ValueError('explicit 32..640 by 32..360 main crop required')
        if type(self.duration_s) is not int or not 2<=self.duration_s<=10:
            raise ValueError('duration 2..10 seconds required')
        if type(self.countdown_s) is not int or not 1<=self.countdown_s<=5:
            raise ValueError('countdown 1..5 seconds required')


def run_capture(factory,output,spec,*,clock=time.monotonic_ns,metrics=system_metrics,
                progress=lambda **kw:None,stopped=lambda:False):
    if not isinstance(spec,MotionSpec):raise ValueError('validated MotionSpec required')
    budget=RunBudget(output,max_bytes=80_000_000,reserve_bytes=1_000_000_000)
    camera=None;error=None;cleanup=[];count=0;last_frame=None;ready=0;labels=set()
    begin=clock();countdown=None;record_start=None;next_health=0;gaps=[]
    try:
        camera=factory();requested,actual=configure_lab(camera)
        wanted=spec.trial.controls()
        if set(wanted)-set(camera.camera_controls):raise ValueError('required controls unavailable')
        setup=setup_record(camera,requested,actual)
        setup.update({'schema_version':1,'motion_spec':asdict(spec),'requested_controls':wanted,
            'focus':'manual candidate, not road calibrated','speed_source':'NOT MEASURED DURING CAPTURE',
            'format':'per-frame uncompressed post-ISP Y .npy crop; NOT sensor RAW',
            'source_motion_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'max_output_bytes':budget.max_bytes,'max_frames':320,'countdown_recorded':False})
        budget.write_json('setup.json',setup)
        camera.set_controls(wanted);sent=clock();camera.start()
        progress(phase='warming',remaining_s=5)
        with budget.open_log() as log:
            while count<320:
                if stopped():raise InterruptedError('user stopped motion service')
                if clock()-begin>(spec.duration_s+spec.countdown_s+8)*1_000_000_000:
                    raise TimeoutError('motion experiment deadline')
                req=None
                try:
                    req=camera.wait(camera.capture_request(wait=False,flush=False),timeout=2)
                    md=req.get_metadata();now=clock();f=frame_from_metadata(md)
                    ag=camera.camera_controls['AnalogueGain']
                    if not ag[0]-.02<=f.analogue_gain<=ag[1]+.02 or f.exposure_us>f.frame_duration_us:
                        raise ValueError('exposure/gain outside actual camera bounds')
                    if not 0<=now-f.exposure_start_ns<=500_000_000 or (
                            last_frame is not None and f.frame_id<=last_frame):
                        raise ValueError('stale or duplicate camera timestamp')
                    previous=last_frame;last_frame=f.frame_id
                    ok,label=check_readback(md,spec.trial)
                    if countdown is None:
                        ready=ready+1 if ok and md['AeState']==2 and f.exposure_start_ns>sent else 0
                        if now-sent>=2_000_000_000 and ready>=3:
                            countdown=now
                        elif now-sent>5_000_000_000:raise TimeoutError('readback failed: '+label)
                        else:continue
                    if not ok:raise ValueError('controls drifted: '+label)
                    if now>=next_health:
                        health=metrics()
                        if health.get('throttled')!='throttled=0x0':raise RuntimeError('power status not clean')
                        next_health=now+2_000_000_000
                    if now-countdown<spec.countdown_s*1_000_000_000:
                        progress(phase='countdown',remaining_s=spec.countdown_s-(now-countdown)/1e9)
                        continue
                    if record_start is None:record_start=now
                    elapsed=(now-record_start)/1e9
                    if elapsed>=spec.duration_s:break
                    progress(phase='recording',remaining_s=spec.duration_s-elapsed)
                    plane=visible_y_plane(req.make_array('main'),actual['main'])
                    x0,y0,x1,y1=spec.roi
                    crop=plane[y0:y1,x0:x1].copy()
                finally:
                    if req is not None:req.release()
                # Write only after releasing camera buffers. Disk slowdown is
                # measured in sensor gaps, never concealed by invented frames.
                name=f'frame_{count:04d}.npy';buf=BytesIO();np.save(buf,crop,allow_pickle=False)
                budget.write_new(name,buf.getvalue())
                gap=None if count==0 else (f.frame_id-previous)/1e6
                if gap is not None:gaps.append(gap)
                labels.add(label)
                budget.append_record(log,{'frame':count,'file':name,'elapsed_s':elapsed,'observed_ns':now,
                    'sensor_gap_ms':gap,'actual_metadata':{k:md.get(k) for k in METADATA_FIELDS},
                    'readback':label,'system':health,'crop_mean_y':float(crop.mean())})
                count+=1
            if count==320:raise RuntimeError('frame cap reached before duration ended')
    except BaseException as failure:
        error=type(failure).__name__+': '+str(failure);raise
    finally:
        if camera is not None:
            for method in ('stop','close'):
                try:getattr(camera,method)()
                except BaseException as failure:cleanup.append(method+': '+str(failure))
        result={'frames':count,'error':error,'cleanup_errors':cleanup,'readback_labels':sorted(labels),
            'max_sensor_gap_ms':max(gaps) if gaps else None,'gaps_over_50ms':sum(g>50 for g in gaps),
            'system_after':metrics(),'available_storage_bytes':budget.available_bytes()}
        budget.write_json('summary.json',result)
        if cleanup and error is None:raise RuntimeError('camera cleanup failed')
    return result


PAGE='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Тест движения</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:22px system-ui;background:#111923;color:#eef3fa;max-width:950px;margin:35px auto;padding:20px}button{font:24px system-ui;padding:20px}#phase{font-size:32px;color:#ffcc80}img{max-width:100%}</style>
<h1>Проверка движения карточки</h1><p>Камеру и освещение не меняйте. Карточку двигайте по полу параллельно рулетке, в сторону свободного пола и обратно примерно на 10–15 см, не поднимая и не поворачивая.</p>
<p>Нажмите кнопку. Во время подготовки и отсчёта карточка неподвижна. После сигнала «ДВИГАЙТЕ» перемещайте её до команды «СТОП». Для каждой серии повторите этот же маршрут.</p>
<img src="/reference.jpg" alt="Область записи"><p id="phase">Подключение…</p><p id="mode"></p><button id="go" disabled onclick="start()">Начать следующую серию</button>
<script>const token='TOKEN';let busy=false;
async function start(){document.getElementById('go').disabled=true;await fetch('/start',{method:'POST',headers:{'X-Trial-Token':token}});}
async function poll(){try{let s=await(await fetch('/status',{cache:'no-store'})).json();
let text={idle:'Готово к следующей серии',warming:'ПОДГОТОВКА — не двигайте',countdown:'НЕ ДВИГАЙТЕ — отсчёт',recording:'ДВИГАЙТЕ КАРТОЧКУ',done:'СТОП — серия сохранена',complete:'СТОП — все серии сохранены',error:'ОШИБКА — остановитесь'};
document.getElementById('phase').textContent=(text[s.phase]||s.phase)+(s.remaining_s>0?' · '+Math.ceil(s.remaining_s)+' с':'');
document.getElementById('mode').textContent='Сохранено '+s.completed+' из '+s.total+(s.error?' · '+s.error:'');
document.getElementById('go').disabled=!['idle','done'].includes(s.phase)||s.completed>=s.total;
}catch(e){document.getElementById('phase').textContent='СВЯЗЬ ПОТЕРЯНА — остановитесь';document.getElementById('go').disabled=true;}setTimeout(poll,250);}poll();</script></html>'''


def enqueue_trial(state,jobs,lock):
    with lock:
        if state['phase'] not in ('idle','done') or state['completed']>=state['total']:
            return False
        jobs.put_nowait(state['completed'])
        state.update(phase='warming',remaining_s=5)
        return True


def serve(args):
    from picamera2 import Picamera2
    root=args.output;root.mkdir(exist_ok=False)
    stop=threading.Event();jobs=queue.Queue(maxsize=1);lock=threading.Lock();token=secrets.token_hex(16)
    schedule=['native-short','shutter-auto-gain','native-normal']
    state={'phase':'idle','remaining_s':0,'completed':0,'total':len(schedule),'error':None}
    def progress(**values):
        with lock:state.update(values)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def setup(self):super().setup();self.connection.settimeout(3)
        def send(self,data,kind='application/json',code=200):
            self.send_response(code);self.send_header('Content-Type',kind)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/':self.send(PAGE.replace('TOKEN',token).encode(),'text/html; charset=utf-8')
            elif self.path=='/status':
                with lock:data=json.dumps(state).encode()
                self.send(data)
            elif self.path=='/reference.jpg':self.send(args.reference.read_bytes(),'image/jpeg')
            else:self.send(b'{}',code=404)
        def do_POST(self):
            if self.path!='/start' or self.headers.get('X-Trial-Token')!=token:
                self.send(b'{}',code=403);return
            self.send(b'{}',code=200 if enqueue_trial(state,jobs,lock) else 409)
    class Server(ThreadingHTTPServer):
        daemon_threads=True
        slots=threading.BoundedSemaphore(8)
        def process_request(self,request,address):
            if not self.slots.acquire(blocking=False):
                self.shutdown_request(request);return
            try:super().process_request(request,address)
            except BaseException:self.slots.release();raise
        def process_request_thread(self,request,address):
            try:super().process_request_thread(request,address)
            finally:self.slots.release()
    server=Server((args.bind,args.port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    start=time.monotonic()
    with (root/'schedule.json').open('x') as f:json.dump({'modes':schedule,'roi':args.roi,'purpose':'feasibility, not paired proof'},f)
    try:
        while not stop.is_set() and time.monotonic()-start<1200:
            try:index=jobs.get(timeout=.5)
            except queue.Empty:continue
            spec=MotionSpec(TrialSpec(schedule[index],(0,0,8,8),(8,0,16,8),tuple(args.colour_gains),args.lens,4000),tuple(args.roi))
            try:
                run_capture(lambda:Picamera2(0),root/f'trial_{index:02d}',spec,progress=progress,stopped=stop.is_set)
                progress(completed=index+1,phase='complete' if index+1==len(schedule) else 'done',remaining_s=0)
            except BaseException as failure:
                progress(phase='error',remaining_s=0,error=type(failure).__name__+': '+str(failure))
    finally:server.shutdown();server.server_close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serve',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--bind',default='127.0.0.1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--roi',nargs=4,type=int);p.add_argument('--colour-gains',nargs=2,type=float)
    p.add_argument('--lens',type=float);p.add_argument('--reference',type=Path)
    a=p.parse_args()
    if not a.serve:print(json.dumps({'dry_run':True,'camera_started':False,'max_trial_bytes':80_000_000,'trials':3}));return
    ip=ipaddress.ip_address(a.bind)
    if ip.version!=4 or not ip.is_private or ip.is_unspecified or not 1024<=a.port<=65535:p.error('private IPv4 and unprivileged port required')
    if any(v is None for v in (a.output,a.roi,a.colour_gains,a.lens,a.reference)):p.error('all measured parameters and reference required')
    MotionSpec(TrialSpec('native-short',(0,0,8,8),(8,0,16,8),tuple(a.colour_gains),a.lens),tuple(a.roi))
    if not a.reference.is_file() or a.reference.stat().st_size>3_000_000:p.error('bounded reference JPEG required')
    serve(a)


if __name__=='__main__':main()
