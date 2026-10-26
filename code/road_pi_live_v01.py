"""Explicitly started local MJPEG demo with v02 exposure controller; no installation.

Stationary profile allows 32.68 ms. Road-lab caps at 4 ms. Neither is road calibrated.
Default dry-run. Run expires after four hours unless explicitly configured otherwise.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import ipaddress
import json
from pathlib import Path
import signal
import threading
import time

from road_core_v02 import ExposureController, SpeedSample
from road_live_http_v01 import LiveState, create_server
from road_observer_v01 import RunBudget, json_safe
from road_pi_controls_v01 import capture, configure_lab, setup_record
from road_pi_loop_v02 import verified_limits
from road_pi_observe_v01 import ROOT, load_config, policy_config, system_metrics


def live_policy(config, limits, profile):
    ceiling = 32680 if profile == 'stationary' else 4000
    limits = replace(limits, max_exposure_us=ceiling,
                     exposure_tolerance_us=max(42, limits.exposure_tolerance_us))
    base = policy_config(config)
    settings = replace(base.detect, max_exposure_us=ceiling, preferred_exposure_us=4000)
    return limits, replace(base, version='live-demo-001-'+profile,
                           detect=settings, view=settings, command_timeout_ns=2_000_000_000)


def run(args):
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import Output
    budget = RunBudget(args.output, max_bytes=80_000_000, reserve_bytes=1_000_000_000)
    state = LiveState()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    camera = encoder = server = worker = None
    error = None
    cleanup = []
    samples = sent_count = ack_count = 0
    sink_errors = []
    class Sink(Output):
        def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
            if self.recording and not audio:
                try:
                    state.publish(frame)
                except Exception as failure:
                    if not sink_errors:
                        sink_errors.append(str(failure))
                    state.update({'error': str(failure), 'profile': args.profile})
                    stop.set()
    try:
        config = load_config(ROOT/'configs/pi_observer_lab_v01.json')
        camera = Picamera2(0)
        requested, actual = configure_lab(camera)
        calibration = json.loads(args.calibration.read_text())
        limits = verified_limits(calibration, camera, actual)
        limits, policy = live_policy(config, limits, args.profile)
        if camera.camera_controls['ExposureTime'][1] < limits.max_exposure_us:
            raise ValueError('exposure ceiling unavailable')
        controller = ExposureController(limits, policy)
        initial = {'AeEnable': False, 'AfMode': 0, 'LensPosition': args.lens,
                   'ExposureTime': 4000, 'AnalogueGain': 2.0}
        camera.set_controls(initial)
        setup = setup_record(camera, requested, actual)
        setup.update({'focus': 'manual candidate, NOT calibrated', 'initial_controls': initial,
                      'policy': asdict(policy), 'limits': asdict(limits), 'duration_s': args.duration,
                      'profile': args.profile, 'speed_source': 'manual profile, NO vehicle speed input',
                      'roi': config['laboratory_metering_rectangle'], 'bind': args.bind, 'port': args.port,
                      'source_live_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'calibration_note': 'readback envelope measured to 12ms; extended stationary range monitored, not calibrated',
                      'video_recorded': False})
        budget.write_json('setup.json', setup)
        camera.start()
        for _ in range(18):
            capture(camera, actual, config)
        encoder = MJPEGEncoder(qp=5, framerate=30)
        camera.start_encoder(encoder, Sink(), name='main')
        server = create_server((args.bind, args.port), state)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        start = time.monotonic()
        next_log = next_system = 0
        metrics = {}
        with budget.open_log() as log:
            while not stop.is_set() and time.monotonic()-start < args.duration:
                frame, scene, record = capture(camera, actual, config)
                now = time.monotonic_ns()
                decision = controller.update(scene, frame, SpeedSample(8., now), now)
                record.update({'profile': args.profile, 'decision': decision.to_record(),
                               'command_dispatch': None, 'elapsed_s': time.monotonic()-start})
                if decision.observed_command_id is not None:
                    ack_count += 1
                if decision.command is not None:
                    cmd = decision.command
                    if not (100 <= cmd.exposure_us <= limits.max_exposure_us and 2 <= cmd.analogue_gain <= 16):
                        raise ValueError('command outside profile limits')
                    dispatch = time.monotonic_ns()
                    camera.set_controls({'AeEnable': False, 'ExposureTime': cmd.exposure_us,
                                         'AnalogueGain': cmd.analogue_gain})
                    controller.mark_sent(cmd.command_id, dispatch)
                    record['command_dispatch'] = {'exposure_us': cmd.exposure_us, 'gain': cmd.analogue_gain,
                                                   'sent_ns': dispatch, 'id': cmd.command_id}
                    sent_count += 1
                samples += 1
                elapsed = time.monotonic()-start
                if elapsed >= next_system:
                    metrics = system_metrics()
                    if budget.available_bytes() < 1_000_000_000:
                        raise RuntimeError('storage reserve reached')
                    if metrics.get('throttled') not in ('throttled=0x0', 'unavailable'):
                        raise RuntimeError('power/thermal throttling detected')
                    next_system = elapsed+10
                record.update({'system': metrics, 'commands_sent': sent_count, 'commands_observed': ack_count})
                state.update(json_safe(record))
                if elapsed >= next_log or record['command_dispatch'] or decision.observed_command_id is not None:
                    budget.append_record(log, record)
                    next_log = elapsed+1
                if any(s in decision.statuses for s in ('CONTROL_TIMEOUT', 'INVALID_INPUT', 'STALE_FRAME')):
                    raise RuntimeError('controller stopped: '+str(decision.statuses))
                status = state.status()
                if elapsed > 3 and (status['video_age_s'] is None or status['video_age_s'] > 2):
                    raise RuntimeError('encoder stopped producing fresh frames')
            if sink_errors:
                raise RuntimeError('encoder output: '+sink_errors[0])
    except BaseException as failure:
        error = type(failure).__name__+': '+str(failure)
        raise
    finally:
        state.close()
        if server:
            server.shutdown()
            server.server_close()
        if camera:
            for action in ([lambda: camera.stop_encoder(encoder)] if encoder else [])+[camera.stop, camera.close]:
                try:
                    action()
                except Exception as failure:
                    cleanup.append(str(failure))
        budget.write_json('summary.json', {'error': error, 'cleanup_errors': cleanup,
                         'samples': samples, 'commands_sent': sent_count, 'commands_observed': ack_count,
                         'stream': state.status(), 'system_after': system_metrics()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-camera', action='store_true')
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--profile', choices=('stationary', 'road-lab'), default='stationary')
    parser.add_argument('--lens', type=float, default=0.0)
    parser.add_argument('--duration', type=int, default=14400)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ip = ipaddress.ip_address(args.bind)
    if ip.version != 4 or not ip.is_private or ip.is_unspecified or not 1024 <= args.port <= 65535:
        parser.error('explicit private IPv4 address and unprivileged port required')
    if not 5 <= args.duration <= 28800 or not 0 <= args.lens <= 2:
        parser.error('duration 5..28800 seconds and lens 0..2 required')
    if not args.run_camera:
        print(json.dumps({'dry_run': True, 'camera_started': False, 'profile': args.profile,
                          'bind': args.bind, 'duration_s': args.duration}))
        return
    if args.output is None or args.calibration is None:
        parser.error('new --output and --calibration required')
    run(args)


if __name__ == '__main__':
    main()
