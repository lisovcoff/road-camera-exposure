"""Raspberry Pi 5 / IMX708 adapter and 24s controlled experiment.

Run from the same directory as road_core_v01.py.
All output goes into a new folder. Does NOT modify previous captures.
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

BASE = Path.home() / "camera_tests"
DURATION_S = 24
INTERVAL_S = 0.5
MAIN_SIZE = (1920, 1080)
LORES_SIZE = (320, 180)
EXPOSURE_INITIAL_US = 4000
GAIN_INITIAL = 8.0
METADATA_FIELDS = (
    "SensorTimestamp", "FrameDuration", "ExposureTime", "AnalogueGain",
    "DigitalGain", "LensPosition", "Lux", "AeState", "AfState", "FocusFoM",
)


def speed_simulation(elapsed):
    """Speed is SIMULATED, not measured from the vehicle."""
    if elapsed < 6:
        return 8.0
    if elapsed < 12:
        return 13.0
    if elapsed < 18:
        return 8.0
    return 15.0


def main():
    if not BASE.is_dir():
        raise SystemExit("Missing ~/camera_tests directory")
    if shutil.disk_usage(BASE).free < 1_000_000_000:
        raise SystemExit("Less than 1 GB available; aborting")

    output = BASE / ("controller_v01_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    print("Output directory:", output, flush=True)

    camera = None
    started = False
    try:
        # Explicitly choose conventional sensor mode, not Sensor HDR.
        with IMX708(0) as sensor:
            sensor.set_sensor_hdr_mode(False)

        camera = Picamera2(0)
        configuration = camera.create_video_configuration(
            main={"size": MAIN_SIZE, "format": "RGB888"},
            lores={"size": LORES_SIZE, "format": "YUV420"},
            raw=None,
            display=None,
            queue=False,
            controls={
                "FrameRate": 30,
                "AeEnable": False,
                "ExposureTime": EXPOSURE_INITIAL_US,
                "AnalogueGain": GAIN_INITIAL,
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": 0.30,
            },
        )
        camera.configure(configuration)
        physical_gain_range = camera.camera_controls["AnalogueGain"]
        controller = ExposureController(
            min_gain=max(1.2, float(physical_gain_range[0])),
            max_gain=min(16.0, float(physical_gain_range[1])),
        )
        run_setup = {
            "created_at": datetime.now().astimezone().isoformat(),
            "duration_s": DURATION_S,
            "analysis_interval_s": INTERVAL_S,
            "speed_kmh": "SIMULATED: 8 (0-6s), 13 (6-12s), 8 (12-18s), 15 (18-24s)",
            "actual_vehicle_speed": "NOT CONNECTED",
            "roi_xyxy_lores": [80, 45, 240, 135],
            "roi_is_calibrated_for_road": False,
            "brightness_target_is_calibrated": False,
            "sensor_hdr": "OFF",
            "controls_range": {str(k): str(v) for k, v in camera.camera_controls.items()},
            "camera_configuration": camera.camera_configuration(),
            "warnings": [
                "Prototype only, not for vehicle use or safety decisions",
                "Center ROI can contain non-road objects",
                "Simulated speed changes profile but no physical motion blur is tested",
                "This test does not measure main stream encoded 30fps",
            ],
        }
        with (output / "setup.json").open("x") as f:
            json.dump(run_setup, f, indent=2, default=str, ensure_ascii=False)

        camera.start()
        started = True
        time.sleep(2)
        start = time.monotonic()
        num_samples = 0
        times_ms = []
        captures = [
            ("detect_first", 4.0),
            ("view_first", 10.0),
            ("detect_second", 16.0),
            ("view_second", 22.0),
        ]
        captured = set()
        with (output / "measurements.jsonl").open("x", buffering=1) as logfile:
            while True:
                tick = time.monotonic()
                elapsed = tick - start
                if elapsed >= DURATION_S:
                    break
                speed = speed_simulation(elapsed)
                saved_this_sample = []
                with camera.captured_request(flush=True) as req:
                    lores = req.make_array("lores")
                    md = req.get_metadata()
                    for label, save_after in captures:
                        if elapsed >= save_after and label not in captured:
                            req.save("main", str(output / (label + ".jpg")))
                            captured.add(label)
                            saved_this_sample.append(label)

                t_analysis = time.monotonic()
                stats = analyze_y_plane(lores[:180, :320])
                decision = controller.decide(
                    stats=stats,
                    actual_exposure_us=md["ExposureTime"],
                    actual_gain=md["AnalogueGain"],
                    speed_kmh=speed,
                    now_seconds=elapsed,
                )
                analysis_ms = (time.monotonic() - t_analysis) * 1000.0
                times_ms.append(analysis_ms)

                # ONLY the adapter knows Picamera2 control names and units.
                camera.set_controls({
                    "AeEnable": False,
                    "ExposureTime": decision["requested_exposure_us"],
                    "AnalogueGain": decision["requested_analogue_gain"],
                })
                record = {
                    "elapsed_s": round(elapsed, 3),
                    "simulated_speed_kmh": speed,
                    "stats": vars(stats),
                    "actual_metadata": {key: md.get(key) for key in METADATA_FIELDS},
                    "unavailable_metadata": [key for key in METADATA_FIELDS if key not in md],
                    "decision": decision,
                    "analysis_ms": round(analysis_ms, 3),
                    "saved_photos": saved_this_sample,
                }
                for label in saved_this_sample:
                    with (output / (label + ".json")).open("x") as photo_log:
                        json.dump(record, photo_log, indent=2, ensure_ascii=False)
                logfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                if num_samples % 4 == 0:
                    print(
                        f"t={elapsed:5.1f}s speed={speed:4.0f} SIMULATED "
                        f"mode={decision['mode']:6s} Ymed={stats.median:5.1f} "
                        f"actual={md['ExposureTime']:5d}us/{md['AnalogueGain']:.2f}x "
                        f"new={decision['requested_exposure_us']:5d}us/"
                        f"{decision['requested_analogue_gain']:.2f}x "
                        f"status={decision['status']}",
                        flush=True,
                    )
                num_samples += 1
                time.sleep(max(0.0, INTERVAL_S - (time.monotonic() - tick)))

        print("Samples:", num_samples)
        print("Saved JPG:", ", ".join(sorted(captured)))
        if times_ms:
            print("Median analysis time (ms):", round(statistics.median(times_ms), 3))
            print("Max analysis time (ms):", round(max(times_ms), 3))
        print("Output directory:", output)
    finally:
        if camera is not None:
            try:
                if started:
                    camera.stop()
            finally:
                camera.close()


if __name__ == "__main__":
    main()
