"""Expanded intermediate-control characterization (v02); default is camera-free dry run.

No HDR changes, network listener, installation or original-file replacement.
Lens position 1.0 is a laboratory hold, NOT a road focus calibration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from road_core_v02 import AnalysisConfig, analyze_y_plane
from road_observer_v01 import RunBudget, frame_from_metadata, json_safe, visible_y_plane
from road_pi_observe_v01 import (ROOT, METADATA_FIELDS, laboratory_zones, load_config,
                                 package_versions, system_metrics)

STEPS = ((2000, 2), (2471, 2), (3039, 2), (3799, 2), (4000, 2),
         (4976, 2), (6215, 2), (7745, 2), (8000, 2), (9999, 2), (12000, 2),
         (4000, 2), (4000, 2.5), (4000, 3.2), (4000, 4.5), (4000, 5.3),
         (4000, 6.4), (4000, 7.5), (4000, 8.7), (4000, 9.9), (4000, 11.1),
         (4000, 12.3), (4000, 14.7), (4000, 15.5), (4000, 16))
FRAMES_PER_STEP = 18


def configure_lab(camera):
    """Same optical/sensor pipeline as observer; explicit manual AE/AF."""
    if "imx708" not in str(camera.camera_properties.get("Model", "")).lower():
        raise ValueError("identified IMX708 required")
    requested = camera.create_video_configuration(
        main={"size": (1920, 1080), "format": "YUV420"},
        lores={"size": (320, 180), "format": "YUV420"}, raw=None,
        display=None, queue=False,
        sensor={"output_size": (2304, 1296), "bit_depth": 10},
        controls={"FrameRate": 30, "AeEnable": False, "ExposureTime": 2000,
                  "AnalogueGain": 2.0, "AfMode": 0, "LensPosition": 1.0})
    requested_copy = json_safe(requested)
    camera.configure(requested)
    actual = camera.camera_configuration()
    for stream, size in (("main", (1920, 1080)), ("lores", (320, 180))):
        if tuple(actual[stream]["size"]) != size or actual[stream]["format"] != "YUV420":
            raise ValueError("unexpected stream configuration")
    if (tuple(actual["sensor"]["output_size"]) != (2304, 1296)
            or actual["sensor"]["bit_depth"] != 10
            or "identity" not in str(actual.get("transform"))):
        raise ValueError("unexpected sensor/transform")
    exp, gain = camera.camera_controls["ExposureTime"], camera.camera_controls["AnalogueGain"]
    if not (exp[0] <= 100 <= 12000 <= exp[1] and gain[0] <= 2 <= 16 <= gain[1]):
        raise ValueError("required bounded controls unavailable")
    return requested_copy, actual


def capture(camera, actual, config, clock=time.monotonic_ns):
    begin = clock()
    req = None
    try:
        req = camera.wait(camera.capture_request(wait=False, flush=False), timeout=1.5)
        returned = clock()
        md = req.get_metadata()
        frame = frame_from_metadata(md)
        plane = visible_y_plane(req.make_array("lores"), actual["lores"])
        stats = analyze_y_plane(plane, laboratory_zones(config),
                               AnalysisConfig(**config["diagnostic_analysis"]),
                               mask_version="module3-lab-grid-001", transform="identity")
        now = clock()
        if not 0 <= now - frame.exposure_start_ns <= 800_000_000:
            raise ValueError("frame clock/age outside laboratory bound")
        record = {"now_ns": now, "capture_started_ns": begin, "capture_returned_ns": returned,
                  "actual_metadata": {k: md.get(k) for k in METADATA_FIELDS},
                  "unavailable_metadata": [k for k in METADATA_FIELDS if k not in md],
                  "exposure_start_ns": frame.exposure_start_ns, "meter_y": stats.meter_y}
        return frame, stats, record
    finally:
        if req is not None:
            req.release()


def characterize(rows):
    """Use settled tail, reject unstable/unobserved transitions; no guessed ACK."""
    results = []
    for step, (exposure, gain) in enumerate(STEPS):
        items = [r for r in rows if r["step"] == step]
        tail = items[-6:]
        stable = len(items) == FRAMES_PER_STEP and len(tail) == 6
        if tail:
            stable = stable and len({r["actual_metadata"]["ExposureTime"] for r in tail}) == 1
            stable = stable and max(r["actual_metadata"]["AnalogueGain"] for r in tail) - min(
                r["actual_metadata"]["AnalogueGain"] for r in tail) < 1e-6
        exp_error = max((abs(r["actual_metadata"]["ExposureTime"] - exposure) for r in tail), default=math.inf)
        gain_error = max((abs(r["actual_metadata"]["AnalogueGain"] - gain) for r in tail), default=math.inf)
        matches = [r for r in items if r["exposure_start_ns"] > r["dispatch_started_ns"]
                   and abs(r["actual_metadata"]["ExposureTime"] - exposure) <= 80
                   and abs(r["actual_metadata"]["AnalogueGain"] - gain) <= .30]
        # Step zero repeats initial controls, so its match is not a transition measurement.
        first = matches[0] if matches and step else None
        results.append({"step": step, "requested_exposure_us": exposure, "requested_gain": gain,
                        "stable": bool(stable), "max_tail_exposure_error_us": exp_error,
                        "max_tail_gain_error": gain_error,
                        "first_matching_sample": None if first is None else first["step_frame"],
                        "dispatch_to_matching_exposure_start_ns": None if first is None else
                            first["exposure_start_ns"] - first["dispatch_started_ns"],
                        "dispatch_to_observation_ns": None if first is None else
                            first["now_ns"] - first["dispatch_started_ns"],
                        "accepted": bool(stable and exp_error <= 80 and gain_error <= .30 and matches)})
    accepted = all(r["accepted"] for r in results)
    return {"schema_version": 1, "accepted": accepted, "steps": results,
            "exposure_tolerance_us": None if not accepted else math.ceil(max(r["max_tail_exposure_error_us"] for r in results)) + 2,
            "gain_tolerance": None if not accepted else max(r["max_tail_gain_error"] for r in results) + .01,
            "scope": "empirical lab envelope for tested points/mode only, not universal sensor quantization"}


def setup_record(camera, requested, actual):
    return {"requested_camera_configuration": requested, "actual_camera_configuration": actual,
            "camera_properties": camera.camera_properties, "camera_controls": camera.camera_controls,
            "versions": package_versions(), "system_before": system_metrics(),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__), Path(__file__).with_name("road_core_v02.py"),
                 Path(__file__).with_name("road_observer_v01.py"), Path(__file__).with_name("road_pi_observe_v01.py"))},
            "road_calibrated": False, "sensor_hdr_state": "not explicitly changed or independently verified",
            "focus": "manual position 1.0, NOT calibrated", "speed_source": "NOT CONNECTED"}


def run(camera_factory, output, clock=time.monotonic_ns):
    budget = RunBudget(output, max_bytes=5_000_000, reserve_bytes=1_000_000_000)
    camera, rows, error, cleanup = None, [], None, []
    cpu_start = time.process_time_ns()
    calibration = None
    try:
        camera = camera_factory()
        config = load_config(ROOT / "configs/pi_observer_lab_v01.json")
        requested, actual = configure_lab(camera)
        setup = setup_record(camera, requested, actual)
        setup.update({"steps": STEPS, "frames_per_step": FRAMES_PER_STEP,
                      "max_loop_s": 30, "max_output_bytes": budget.max_bytes})
        budget.write_json("setup.json", setup)
        camera.start()
        start = clock()
        for _ in range(6):
            capture(camera, actual, config, clock)
        with budget.open_log() as log:
            for step, (exposure, gain) in enumerate(STEPS):
                sent = clock()
                camera.set_controls({"AeEnable": False, "ExposureTime": exposure, "AnalogueGain": float(gain)})
                dispatched = clock()
                for index in range(FRAMES_PER_STEP):
                    if clock() - start > 30_000_000_000:
                        raise TimeoutError("30 second characterization budget exceeded")
                    _, _, record = capture(camera, actual, config, clock)
                    record.update({"step": step, "step_frame": index,
                        "requested": {"ExposureTime": exposure, "AnalogueGain": gain, "AeEnable": False},
                        "dispatch_started_ns": sent, "dispatch_returned_ns": dispatched,
                        "available_storage_bytes": budget.available_bytes()})
                    budget.append_record(log, record)
                    rows.append(record)
        calibration = characterize(rows)
        calibration["camera_model"] = str(camera.camera_properties["Model"])
        calibration["actual_camera_configuration"] = json_safe(actual)
        calibration["versions"] = setup["versions"]
        budget.write_json("calibration.json", calibration)
        if not calibration["accepted"]:
            raise ValueError("control characterization rejected; do not enable closed loop")
    except BaseException as failure:
        error = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        if camera is not None:
            for method in ("stop", "close"):
                try:
                    getattr(camera, method)()
                except BaseException as failure:
                    cleanup.append(method + ": " + str(failure))
        summary = {"samples": len(rows), "accepted": calibration is not None and calibration["accepted"],
                   "error": error, "cleanup_errors": cleanup, "system_after": system_metrics(),
                   "process_cpu_ns": time.process_time_ns() - cpu_start,
                   "available_storage_bytes": budget.available_bytes()}
        budget.write_json("summary.json", summary)
        if cleanup and error is None:
            raise RuntimeError("camera cleanup failed")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-camera", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.run_camera:
        print(json.dumps({"dry_run": True, "camera_started": False, "steps": STEPS,
                          "frames": 6 + len(STEPS) * FRAMES_PER_STEP, "output_budget_bytes": 5_000_000}))
        return 0
    if args.output is None:
        parser.error("a NEW --output directory is required")
    from picamera2 import Picamera2
    print(json.dumps(json_safe(run(lambda: Picamera2(0), args.output))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
