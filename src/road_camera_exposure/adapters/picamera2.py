"""Picamera2 target-motion demo with native gain, colour preview and validation capture.

Explicit fixed target ROI, not a calibrated road installation. Native gain/ISP.
Default dry-run, no package or system configuration changes.
"""
import argparse
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

from ..colour import preview_planes
from ..motion import TargetTracker,ShutterPolicy,ControlGate
from ..web import LiveState,create_server
from ..imaging import RunBudget,frame_from_metadata,json_safe,visible_y_plane
from .picamera2_support import configure_lab,setup_record
from .picamera2_support import METADATA_FIELDS,system_metrics
from .picamera2_support import TrialSpec,check_readback


PAGE='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Адаптация к движению</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{background:#111923;color:#eef3fa;font:20px system-ui;max-width:1100px;margin:25px auto;padding:15px}img{width:100%;max-height:65vh;object-fit:contain}button{padding:16px;font-size:20px}#phase{color:#ffcc80;font-size:28px}</style>
<h1>Автоматическая выдержка по движению</h1><img src="/stream.mjpg"><p id="stats"></p><p id="phase"></p>
<button id="record" onclick="record()">Записать проверку: остановка → движение → остановка</button>
<p>После нажатия: 5 с подготовки, затем 3 с не двигать, 6 с двигать карточку по прежнему маршруту, 6 с не двигать. Следите за командами.</p>
<p>Лабораторный прототип. Поиск движения ограничен карточкой. Цветной просмотр, расчёт по яркости. Калибровка для дорожной установки требуется.</p>
<script>const token='TOKEN';async function record(){document.getElementById('record').disabled=true;await fetch('/record',{method:'POST',headers:{'X-Record-Token':token}})}
async function poll(){try{let s=await(await fetch('/status.json',{cache:'no-store'})).json();let m=s.actual_metadata||{},d=s.motion||{};
document.getElementById('stats').textContent=`${s.policy?.mode||'Подготовка'} · Выдержка ${((m.ExposureTime||0)/1000).toFixed(2)} мс · Усиление ${(m.AnalogueGain||0).toFixed(2)}× · Скорость ${d.speed_px_s==null?'не определена':d.speed_px_s.toFixed(0)+' пикс/с'} · ${s.error||''}`;
document.getElementById('phase').textContent=s.instruction||'Камера работает. Для проверки нажмите кнопку.';document.getElementById('record').disabled=s.record_state!=='idle';
}catch(e){document.getElementById('phase').textContent='Связь потеряна — остановитесь';}setTimeout(poll,250);}poll();</script></html>'''.encode('utf-8')


def run(args):
    from picamera2 import Picamera2
    budget=RunBudget(args.output,max_bytes=args.session_bytes,reserve_bytes=args.reserve_bytes)
    state=LiveState();jobs=queue.Queue(maxsize=1);record_lock=threading.Lock();record_state='idle'
    token=secrets.token_hex(16);stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    camera=server=None;validation=None;vlog=None;error=None;cleanup=[];count=0;vc=0
    try:
        camera=Picamera2(args.camera_index);requested,actual=configure_lab(camera)
        spec=TrialSpec('native-normal',(0,0,8,8),(8,0,16,8),tuple(args.colour_gains),args.lens)
        controls=spec.controls();camera.set_controls(controls)
        setup=setup_record(camera,requested,actual)
        setup.update({'purpose':'LAB_TARGET_ADAPTIVE_NOT_ROAD_CALIBRATED','initial_controls':controls,
                      'roi':args.roi,'tracking_rectangle':args.tracking_rectangle,'fixed_focus':args.lens,
                      'speed_source':'target affine motion and exposure-midpoint timestamps',
                      'record_protocol':'3s still / 6s moving / 6s still',
                      'motion_policy':'0.8px heuristic; 20% speed margin; 25px/s additive; 1..8ms; 1s stillness for native',
                      'preview':'real cropped YUV420 colour JPEG; analysis uses unchanged Y','duration_s':args.duration})
        budget.write_json('setup.json',setup)
        camera.start();warm=time.monotonic();matched=0
        gate=ControlGate();tracker=TargetTracker(rectangle=tuple(args.tracking_rectangle));policy=ShutterPolicy()
        while time.monotonic()-warm<5:
            req=camera.wait(camera.capture_request(wait=False,flush=False),timeout=2)
            try:
                md=req.get_metadata();ok,_=check_readback(md,spec)
                if md.get('ExposureTimeMode') is not None:gate.mode_cache=md['ExposureTimeMode'];gate.mode_observed_ns=time.monotonic_ns()
                matched=matched+1 if ok and md.get('AeState')==2 else 0
            finally:req.release()
            if matched>=3 and time.monotonic()-warm>=2:break
        if matched<3:raise TimeoutError('initial native AE/focus/AWB did not settle')
        server=create_server((args.bind,args.port),state);base=server.RequestHandlerClass
        class Handler(base):
            def do_GET(self):
                if self.path=='/':self.response(PAGE.replace(b'TOKEN',token.encode()),'text/html; charset=utf-8')
                else:super().do_GET()
            def do_POST(self):
                nonlocal record_state
                with record_lock:
                    if self.path!='/record' or self.headers.get('X-Record-Token')!=token:
                        self.response(b'{}','application/json',403);return
                    if record_state!='idle':self.response(b'{}','application/json',409);return
                    record_state='armed';jobs.put_nowait(time.monotonic())
                self.response(b'{}','application/json')
        server.RequestHandlerClass=Handler
        threading.Thread(target=server.serve_forever,daemon=True).start()
        started=time.monotonic();next_track=next_preview=next_log=next_health=0
        motion={'speed_px_s':None,'valid':False};desired={'mode':'UNKNOWN','exposure_us':4000};armed=None;record_start=None
        last_sensor=None;health={}
        with budget.open_log() as log:
            while not stop.is_set() and time.monotonic()-started<args.duration:
                req=None
                try:
                    req=camera.wait(camera.capture_request(wait=False,flush=False),timeout=2)
                    md=req.get_metadata();now=time.monotonic_ns();frame=frame_from_metadata(md);wall=time.monotonic()
                    if not 0<=now-frame.exposure_start_ns<=500_000_000 or (last_sensor is not None and frame.frame_id<=last_sensor):
                        raise ValueError('stale or duplicate frame')
                    last_sensor=frame.frame_id
                    main_array=req.make_array('main')
                    plane=visible_y_plane(main_array,actual['main']);x0,y0,x1,y1=args.roi
                    crop=plane[y0:y1,x0:x1].copy()
                finally:
                    if req is not None:req.release()
                if wall>=next_health:
                    health=system_metrics()
                    if health.get('throttled')!='throttled=0x0' or budget.available_bytes()<args.reserve_bytes:
                        raise RuntimeError('power/storage condition not clean')
                    next_health=wall+5
                if wall>=next_track:
                    compute=time.monotonic_ns();motion=tracker.observe(crop,md['SensorTimestamp']-md['ExposureTime']*500)
                    motion['compute_ms']=(time.monotonic_ns()-compute)/1e6
                    desired=policy.update(motion['speed_px_s'],motion['valid'],now);next_track=wall+.12
                command=gate.update(desired,md,now)
                if command is not None:
                    sent=time.monotonic_ns();camera.set_controls(command);gate.sent(command,sent)
                try:armed=jobs.get_nowait()
                except queue.Empty:pass
                instruction=''
                if armed is not None and validation is None:
                    remain=5-(wall-armed)
                    instruction='ПОДГОТОВКА — не двигайте · '+str(max(0,int(np.ceil(remain))))+' с'
                    if remain<=0:
                        validation=RunBudget(args.output/'validation_001',max_bytes=args.validation_bytes,reserve_bytes=args.reserve_bytes)
                        validation.write_json('setup.json',{**setup,'mode':'adaptive-motion','motion_spec':{'roi':args.roi,'trial':{'mode':'adaptive-motion'}},'source':'actual closed-loop controls, not replay'})
                        vlog=validation.open_log();record_start=wall
                        with record_lock:record_state='recording'
                elapsed=None if record_start is None else wall-record_start
                if record_state=='recording':
                    if elapsed<3:instruction='НЕ ДВИГАЙТЕ · '+str(int(np.ceil(3-elapsed)))+' с'
                    elif elapsed<9:instruction='ДВИГАЙТЕ КАРТОЧКУ · '+str(int(np.ceil(9-elapsed)))+' с'
                    elif elapsed<15:instruction='СТОП — держите неподвижно · '+str(int(np.ceil(15-elapsed)))+' с'
                    else:
                        vlog.close();vlog=None
                        validation.write_json('summary.json',{'frames':vc,'error':None,'cleanup_errors':[],
                            'commands_sent':gate.sent_count,'commands_observed':gate.observed_count})
                        with record_lock:record_state='complete'
                if record_state=='complete':instruction='Проверка сохранена. Алгоритм продолжает работать.'
                record={'elapsed_s':wall-started,'actual_metadata':{k:md.get(k) for k in METADATA_FIELDS},
                    'motion':motion,'policy':desired,'command_dispatch':command,'pending':gate.pending,
                    'commands_sent':gate.sent_count,'commands_observed':gate.observed_count,'system':health,
                    'record_state':record_state,'instruction':instruction,'profile':'adaptive-target-lab'}
                if record_state=='recording':
                    if vc>=480:raise RuntimeError('validation frame cap exceeded')
                    filename=f'frame_{vc:04d}.npy';buffer=BytesIO();np.save(buffer,crop,allow_pickle=False)
                    validation.write_new(filename,buffer.getvalue())
                    import simplejpeg
                    py,pu,pv=preview_planes(main_array,actual['main'],args.roi,rotate=args.rotate_preview)
                    colour_name=f'colour_{vc:04d}.jpg'
                    validation.write_new(colour_name,simplejpeg.encode_jpeg_yuv_planes(py,pu,pv,args.preview_quality))
                    validation.append_record(vlog,{**record,'frame':vc,'file':filename,'colour_file':colour_name,'validation_elapsed_s':elapsed});vc+=1
                if wall>=next_preview:
                    import simplejpeg
                    py,pu,pv=preview_planes(main_array,actual['main'],args.roi,rotate=args.rotate_preview)
                    state.publish(simplejpeg.encode_jpeg_yuv_planes(py,pu,pv,args.preview_quality));next_preview=wall+.1
                state.update(json_safe(record))
                if wall>=next_log or command:
                    budget.append_record(log,record);next_log=wall+.5
                count+=1
    except BaseException as failure:error=type(failure).__name__+': '+str(failure);raise
    finally:
        if vlog is not None:
            vlog.close()
            validation.write_json('summary.json',{'frames':vc,'error':error or 'interrupted before completion','cleanup_errors':[]})
        state.close()
        if server:server.shutdown();server.server_close()
        if camera:
            for method in ('stop','close'):
                try:getattr(camera,method)()
                except BaseException as failure:cleanup.append(str(failure))
        budget.write_json('summary.json',{'error':error,'cleanup_errors':cleanup,'samples':count,'system_after':system_metrics()})
