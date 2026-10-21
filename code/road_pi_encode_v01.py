"""Bounded H264/MJPEG sink benchmark, optional lab exposure controller.

No video file or network. Retains at most six packets / 3 MB in RAM for a
decode smoke check AFTER performance measurement. Default is dry-run.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time

from road_core_v02 import ExposureController, SpeedSample
from road_observer_v01 import RunBudget, json_safe
from road_pi_controls_v01 import capture, configure_lab, setup_record
from road_pi_loop_v02 import verified_limits, simulated_speed
from road_pi_observe_v01 import ROOT, load_config, policy_config, system_metrics


class PacketStats:
    """Bounded metadata/first-packet storage. Limit failure signals main thread."""
    def __init__(self, max_packets=600, max_bytes=100_000_000):
        self.lock = threading.Lock()
        self.max_packets, self.max_bytes = max_packets, max_bytes
        self.rows, self.payloads, self.bytes, self.retained, self.error = [], [], 0, 0, None

    def add(self, frame, timestamp, keyframe, now):
        with self.lock:
            if self.error:
                return
            if (type(timestamp) is not int or timestamp < 0 or len(self.rows) >= self.max_packets
                    or len(frame) > 3_000_000 or self.bytes + len(frame) > self.max_bytes):
                self.error = "packet metadata/byte budget exceeded or invalid timestamp"
                return
            if self.rows and timestamp <= self.rows[-1]["pts_us"]:
                self.error = "non-increasing encoded timestamps"
                return
            self.rows.append({"pts_us": timestamp, "output_ns": now, "bytes": len(frame), "keyframe": bool(keyframe)})
            self.bytes += len(frame)
            if len(self.payloads) < 6 and self.retained + len(frame) <= 3_000_000:
                self.payloads.append(bytes(frame))
                self.retained += len(frame)

    def check(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)

    def result(self, first_timestamp_us):
        with self.lock:
            rows = list(self.rows)
        pts = [r["pts_us"] for r in rows]
        arrivals = [r["output_ns"] for r in rows]
        span = (pts[-1] - pts[0]) / 1e6 if len(pts) >= 2 else 0
        age = ([] if first_timestamp_us is None else
               [(r["output_ns"] - (first_timestamp_us + r["pts_us"]) * 1000) / 1e6 for r in rows])
        def bounds(values):
            return None if not values else {"min": min(values), "median": statistics.median(values), "max": max(values)}
        return {"packets": len(rows), "encoded_bytes": self.bytes, "error": self.error,
                "packet_rate_by_pts": None if not span else (len(pts)-1)/span,
                "packet_rate_by_arrival": None if len(arrivals)<2 or arrivals[-1]==arrivals[0] else
                    (len(arrivals)-1)*1e9/(arrivals[-1]-arrivals[0]),
                "pts_gaps_us": bounds([b-a for a,b in zip(pts,pts[1:])]),
                "arrival_gaps_ms": bounds([(b-a)/1e6 for a,b in zip(arrivals,arrivals[1:])]),
                "readout_to_sink_ms": bounds(age),
                "mean_encoded_mbps_by_pts": None if not span else self.bytes*8/span/1e6,
                "retained_payload_bytes": self.retained, "rows": rows}


def make_encoder(codec, stats):
    from picamera2.encoders import H264Encoder, MJPEGEncoder
    from picamera2.outputs import Output
    class Sink(Output):
        def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
            if self.recording and not audio:
                stats.add(frame, timestamp, keyframe, time.monotonic_ns())
    if codec == "h264":
        encoder = H264Encoder(bitrate=8_000_000, repeat=True, iperiod=30, framerate=30)
        encoder.threads = 2
    else:
        encoder = MJPEGEncoder(qp=5, framerate=30)
    return encoder, Sink()


def decode_smoke(codec, payloads):
    import av
    decoder = av.CodecContext.create(codec, "r")
    frames = []
    for data in payloads:
        frames.extend(decoder.decode(av.Packet(data)))
    frames.extend(decoder.decode(None))
    return {"packets_tested":len(payloads), "frames_decoded":len(frames),
            "sizes":sorted({(f.width,f.height) for f in frames}),
            "scope":"only first bounded packets; no road-detail/compression-quality validation"}


def run(camera_factory, codec, output, calibration_path=None, encoder_factory=make_encoder,
        decoder=decode_smoke, clock=time.monotonic_ns):
    if codec not in ("h264","mjpeg"):
        raise ValueError("unsupported codec")
    calibration_bytes = None if calibration_path is None else Path(calibration_path).read_bytes()
    calibration = None if calibration_bytes is None else json.loads(calibration_bytes)
    budget = RunBudget(output,max_bytes=5_000_000,reserve_bytes=1_000_000_000)
    stats = PacketStats()
    camera, encoder, error, cleanup = None, None, None, []
    samples, sent_count, observed_count, pending = 0, 0, 0, None
    duration = 8 if calibration is None else 16
    cpu, wall = time.process_time_ns(), clock()
    perf_cpu = perf_wall = None
    start = None
    try:
        camera = camera_factory()
        config = load_config(ROOT/"configs/pi_observer_lab_v01.json")
        requested, actual = configure_lab(camera)
        controller = None
        if calibration is not None:
            limits = verified_limits(calibration,camera,actual)
            policy = policy_config(config)
            policy = replace(policy,version="pi-encoded-loop-lab-001",view=replace(policy.view,max_exposure_us=8000))
            controller = ExposureController(limits,policy)
        setup = setup_record(camera,requested,actual)
        setup.update({"codec":codec,"duration_s":duration,"video_written":False,
            "settings":{"h264_bitrate":8_000_000,"h264_threads":2,"mjpeg_qp":5},
            "controller_enabled":controller is not None,
            "calibration_sha256": None if calibration_bytes is None else hashlib.sha256(calibration_bytes).hexdigest(),
            "source_encoder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "manual_override_before_start":None if controller else {"ExposureTime":4000,"AnalogueGain":16.,"AeEnable":False}})
        budget.write_json("setup.json",setup)
        if controller is None:
            camera.set_controls({"AeEnable":False,"ExposureTime":4000,"AnalogueGain":16.})
        camera.start()
        for _ in range(18): capture(camera,actual,config,clock)
        encoder,sink = encoder_factory(codec,stats)
        camera.start_encoder(encoder,sink,name="main")
        budget.write_json("encoder.json", {"class":type(encoder).__module__+"."+type(encoder).__name__,
            "bitrate":getattr(encoder,"bitrate",None),"qp":getattr(encoder,"qp",None),
            "threads":getattr(encoder,"threads","library-controlled"),
            "framerate":getattr(encoder,"framerate",None),
            "settings_note":"MJPEG library uses its internal thread count; thread attribute may be absent"})
        start = clock()
        next_metrics = 0
        with budget.open_log() as log:
            while clock()-start < duration*1_000_000_000 and samples < duration*32:
                stats.check()
                frame,scene,record = capture(camera,actual,config,clock)
                now = clock()
                record.update({"sample_index":samples,"elapsed_ns":now-start,"command_dispatch":None})
                decision = None
                if controller:
                    decision=controller.update(scene,frame,SpeedSample(simulated_speed((now-start)/1e9),now),now)
                    record["decision"]=decision.to_record()
                    if decision.observed_command_id is not None: observed_count+=1
                    if decision.command is not None:
                        cmd=decision.command
                        if not (100 <= cmd.exposure_us <= 8000 and 2 <= cmd.analogue_gain <= 16):
                            raise ValueError("command outside lab bounds")
                        sent=clock()
                        camera.set_controls({"AeEnable":False,"ExposureTime":cmd.exposure_us,"AnalogueGain":cmd.analogue_gain})
                        controller.mark_sent(cmd.command_id,sent)
                        record["command_dispatch"]={"id":cmd.command_id,"started_ns":sent,"returned_ns":clock(),
                                                    "exposure_us":cmd.exposure_us,"gain":cmd.analogue_gain}
                        sent_count+=1
                    pending=decision.pending_command_id
                if now-start >= next_metrics:
                    metrics=system_metrics()
                    record["system"]=metrics
                    next_metrics=now-start+4_000_000_000
                    if metrics.get("throttled") not in ("throttled=0x0","unavailable"):
                        raise RuntimeError("power/thermal throttling detected")
                budget.append_record(log,record)
                samples+=1
                if decision and any(s in decision.statuses for s in ("CONTROL_TIMEOUT","INVALID_INPUT","STALE_FRAME")):
                    raise RuntimeError("control loop stopped: "+str(decision.statuses))
    except BaseException as failure:
        error=type(failure).__name__+": "+str(failure)
        raise
    finally:
        if camera is not None:
            actions=[]
            if encoder is not None: actions.append(lambda:camera.stop_encoder(encoder))
            actions.extend((camera.stop,camera.close))
            for action in actions:
                try: action()
                except BaseException as failure: cleanup.append(str(failure))
        perf_cpu,perf_wall=time.process_time_ns()-cpu,clock()-wall
        encoded=stats.result(None if encoder is None else encoder.firsttimestamp)
        budget.write_json("packets.json",encoded)
        decoded=None
        if error is None and not cleanup and stats.error is None:
            try:
                decoded=decoder(codec,stats.payloads)
                if decoded["frames_decoded"] < 1 or decoded["sizes"] != [(1920,1080)]:
                    raise ValueError("decode smoke failed")
            except BaseException as failure:
                error="decode: "+str(failure)
        summary={"codec":codec,"samples":samples,"commands_sent":sent_count,"commands_observed":observed_count,
                 "pending_command_id":pending,"error":error,"cleanup_errors":cleanup,
                 "packet_error":stats.error,"decode_smoke":decoded,
                 "measurement_wall_ns":perf_wall,"process_cpu_ns_before_decode":perf_cpu,
                 "system_after":system_metrics(),"available_storage_bytes":budget.available_bytes(),
                 "note":"CPU/wall includes setup/warmup/encoder drain, excludes post-run decoder; no network/inference"}
        budget.write_json("summary.json",summary)
        if error is not None or cleanup or stats.error:
            if sys.exc_info()[0] is None: raise RuntimeError("encoder experiment failed; inspect new run")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec",choices=("h264","mjpeg"),default="h264")
    parser.add_argument("--run-camera",action="store_true")
    parser.add_argument("--calibration",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    if not args.run_camera:
        print(json.dumps({"dry_run":True,"camera_started":False,"codec":args.codec,
                          "controller_enabled":args.calibration is not None,"duration_s":16 if args.calibration else 8,
                          "disk_budget":5_000_000,"packet_limit":600,"encoded_byte_limit":100_000_000}))
        return 0
    if args.output is None: parser.error("NEW --output required")
    from picamera2 import Picamera2
    print(json.dumps(json_safe(run(lambda:Picamera2(0),args.codec,args.output,args.calibration))))
    return 0


if __name__ == "__main__": sys.exit(main())
