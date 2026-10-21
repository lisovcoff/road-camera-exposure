"""Raspberry Pi Picamera2: 24-s lighting step-response test for road_core_v01.

Manual experiment, NOT for driving or safety-critical use. Simulated speed
remains 8 km/h (DETECT). The user switches a lamp at 8 and 16 seconds.
Existing files are not overwritten. No video, just 3 photos + JSONL log.
"""
from datetime import datetime
from pathlib import Path
import json
import shutil
import statistics
import time

from picamera2 import Picamera2
from picamera2.devices.imx708 import IMX708
from libcamera import controls

from road_core_v01 import ExposureController, analyze_y_plane

BASE = Path.home() / 'camera_tests'
DURATION = 24.0
PERIOD = 0.5
SPEED_SIMULATED = 8.0


def phase(t):
    if t < 8.0:
        return 'DIM_1'
    if t < 16.0:
        return 'BRIGHT'
    return 'DIM_2'


def main():
    if not BASE.is_dir() or shutil.disk_usage(BASE).free < 1_000_000_000:
        raise SystemExit('Missing test directory or less than 1 GB free')
    out = BASE / ('light_response_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    print('Output:', out, flush=True)
    print('Prepare a FIXED camera and a lamp that illuminates its central view.', flush=True)
    print('START: lamp DIM/OFF. At 8 s make it BRIGHT/ON; at 16 s DIM/OFF again.', flush=True)
    print('Do not move the camera or the scene. Simulated speed is always 8 km/h.', flush=True)

    camera = None
    started = False
    try:
        with IMX708(0) as sensor:
            sensor.set_sensor_hdr_mode(False)
        camera = Picamera2(0)
        cfg = camera.create_video_configuration(
            main={'size': (1920, 1080), 'format': 'RGB888'},
            lores={'size': (320, 180), 'format': 'YUV420'},
            raw=None, display=None, queue=False,
            controls={
                'FrameRate': 30, 'AeEnable': False,
                'ExposureTime': 4000, 'AnalogueGain': 8.0,
                'AfMode': controls.AfModeEnum.Manual,
                'LensPosition': 0.3,
            },
        )
        camera.configure(cfg)
        physical_gain = camera.camera_controls['AnalogueGain']
        algorithm = ExposureController(
            min_gain=max(1.2, float(physical_gain[0])),
            max_gain=min(16.0, float(physical_gain[1])),
        )
        setup = {
            'created_at': datetime.now().astimezone().isoformat(),
            'duration_s': DURATION, 'control_interval_s': PERIOD,
            'simulated_speed_kmh': SPEED_SIMULATED,
            'lighting_schedule_requested': [
                {'start_s': 0, 'label': 'DIM_1'},
                {'start_s': 8, 'label': 'BRIGHT'},
                {'start_s': 16, 'label': 'DIM_2'},
            ],
            'lighting_change_is_manual': True,
            'actual_lamp_times_unverified': True,
            'sensor_hdr': 'OFF',
            'roi_xyxy_lores': [80, 45, 240, 135],
            'roi_calibrated_to_road': False,
            'target_brightness_calibrated': False,
            'camera_configuration': camera.camera_configuration(),
            'camera_controls': {str(k): str(v) for k, v in camera.camera_controls.items()},
            'warnings': [
                'Prototype only; not validated for vehicle deployment',
                'Illumination changes must be made by the user',
                'ROI is a temporary rectangle, not the vehicle road area',
                'This does not verify encoded 30 fps',
            ],
        }
        with (out / 'setup.json').open('x', encoding='utf-8') as f:
            json.dump(setup, f, ensure_ascii=False, indent=2, default=str)
        camera.start()
        started = True
        time.sleep(2)
        print('TEST START: DIM/OFF', flush=True)
        start = time.monotonic()
        samples = 0
        analysis_times = []
        saved = set()
        snapshots = [('dim_before', 6.0), ('bright', 14.0), ('dim_after', 22.0)]
        last_phase = None
        with (out / 'measurements.jsonl').open('x', encoding='utf-8', buffering=1) as log:
            while True:
                tick = time.monotonic()
                t = tick - start
                if t >= DURATION:
                    break
                requested_phase = phase(t)
                if requested_phase != last_phase:
                    if requested_phase == 'BRIGHT':
                        print('>>> 8s: SWITCH LAMP TO BRIGHT/ON NOW <<<', flush=True)
                    elif requested_phase == 'DIM_2':
                        print('>>> 16s: SWITCH LAMP BACK TO DIM/OFF NOW <<<', flush=True)
                    last_phase = requested_phase
                photos = []
                with camera.captured_request(flush=True) as req:
                    yuv = req.make_array('lores')
                    md = req.get_metadata()
                    for name, when in snapshots:
                        if t >= when and name not in saved:
                            req.save('main', str(out / (name + '.jpg')))
                            saved.add(name)
                            photos.append(name)
                began = time.monotonic()
                stats = analyze_y_plane(yuv[:180, :320])
                decision = algorithm.decide(
                    stats, md['ExposureTime'], md['AnalogueGain'], SPEED_SIMULATED, t
                )
                elapsed_ms = (time.monotonic() - began) * 1000
                analysis_times.append(elapsed_ms)
                camera.set_controls({
                    'AeEnable': False,
                    'ExposureTime': decision['requested_exposure_us'],
                    'AnalogueGain': decision['requested_analogue_gain'],
                })
                record = {
                    'elapsed_s': round(t, 3),
                    'requested_lighting_phase': requested_phase,
                    'simulated_speed_kmh': SPEED_SIMULATED,
                    'stats': vars(stats),
                    'actual_metadata': {key: md.get(key) for key in (
                        'SensorTimestamp', 'ExposureTime', 'AnalogueGain',
                        'DigitalGain', 'FrameDuration', 'LensPosition', 'Lux')},
                    'decision': decision,
                    'analysis_ms': round(elapsed_ms, 3),
                    'saved_photos': photos,
                }
                log.write(json.dumps(record, ensure_ascii=False) + '\n')
                for name in photos:
                    with (out / (name + '.json')).open('x', encoding='utf-8') as f:
                        json.dump(record, f, ensure_ascii=False, indent=2)
                if samples % 2 == 0:
                    print(f"t={t:4.1f}s phase={requested_phase:6s} "
                          f"Y={stats.median:5.1f} "
                          f"actual={md['ExposureTime']:5d}us/{md['AnalogueGain']:.2f}x "
                          f"next={decision['requested_exposure_us']:5d}us/"
                          f"{decision['requested_analogue_gain']:.2f}x "
                          f"{decision['status']}", flush=True)
                samples += 1
                time.sleep(max(0.0, PERIOD - (time.monotonic() - tick)))
        print('Samples:', samples)
        print('Images:', ', '.join(sorted(saved)))
        print('Median analysis ms:', round(statistics.median(analysis_times), 3))
        print('Output:', out)
    finally:
        if camera is not None:
            try:
                if started:
                    camera.stop()
            finally:
                camera.close()


if __name__ == '__main__':
    main()
