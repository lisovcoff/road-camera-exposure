"""Bounded Picamera2 observation experiment. Default mode is config-only dry run.

No encoder, socket, set_controls, sensor-HDR write or policy dispatch exists in
this script. --run-camera DOES configure/start the camera and write a new run.
"""
import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from road_core_v02 import (AnalysisConfig, ExposureLimits, PolicyConfig, Profile,
                          SpeedSample, ZoneMask, analyze_y_plane)
from road_observer_v01 import RunBudget, ShadowAdvisor, frame_from_metadata, json_safe, visible_y_plane

ROOT = Path(__file__).resolve().parents[1]
METADATA_FIELDS = (
    "SensorTimestamp", "ExposureTime", "AnalogueGain", "DigitalGain", "FrameDuration",
    "LensPosition", "AfState", "FocusFoM", "AeState", "Lux", "ScalerCrop",
    "ColourGains", "ColourTemperature", "HdrChannel", "ExposureTimeMode", "AnalogueGainMode",
)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_config(config)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    if (config.get("schema_version") != 1 or config.get("purpose") != "module3_lab_observation_only"
            or config.get("road_calibrated") is not False or config.get("transform") != "identity"
            or config.get("roi_source") != "new_provisional_laboratory_grid_NOT_dashcam_dataset"):
        raise ValueError("explicit uncalibrated Module 3 lab configuration required")
    if config.get("main_size") != [1920, 1080] or config.get("lores_size") != [320, 180]:
        raise ValueError("this bounded experiment requires 1920x1080 + 320x180")
    if config.get("sensor") != {"output_size": [2304, 1296], "bit_depth": 10}:
        raise ValueError("unexpected requested sensor mode")
    bounds = {"duration_s": (1, 15), "sample_period_s": (0.5, 2), "capture_timeout_s": (0.1, 2),
              "max_output_bytes": (1, 20_000_000), "reserve_bytes": (1_000_000_000, 10**12),
              "snapshot_max_bytes": (1, 3_000_000), "requested_fps": (30, 30)}
    for key, (low, high) in bounds.items():
        value = config[key]
        if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} outside experiment bounds")
    for key in ("max_output_bytes", "reserve_bytes", "snapshot_max_bytes"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be integer bytes")
    snapshots = config["snapshot_at_s"]
    if (not isinstance(snapshots, list) or len(snapshots) > 3
            or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v < config["duration_s"] for v in snapshots)
            or snapshots != sorted(set(snapshots))):
        raise ValueError("at most 3 sorted snapshot times inside experiment")
    speed = config["simulated_speed_kmh"]
    if speed is not None and (type(speed) not in (int, float) or not np.isfinite(speed) or speed < 0):
        raise ValueError("simulated speed must be nonnegative or null")
    policy_config(config)
    AnalysisConfig(**config["diagnostic_analysis"])
    laboratory_zones(config)
    return config


def policy_config(config):
    values = dict(config["policy"])
    values["detect"], values["view"] = Profile(**values["detect"]), Profile(**values["view"])
    return PolicyConfig(**values)


def laboratory_zones(config):
    width, height = config["lores_size"]
    rectangle = config["laboratory_metering_rectangle"]
    if (not isinstance(rectangle, list) or len(rectangle) != 4
            or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in rectangle)):
        raise ValueError("explicit normalized laboratory rectangle required")
    x0, y0, x1, y1 = rectangle
    if x0 >= x1 or y0 >= y1:
        raise ValueError("empty laboratory rectangle")
    edges = np.rint(np.linspace(x0 * width, x1 * width, 4)).astype(int)
    zones = []
    for i, name in enumerate(("left", "center", "right")):
        mask = np.zeros((height, width), dtype=bool)
        mask[round(y0 * height):round(y1 * height), edges[i]:edges[i + 1]] = True
        if mask.sum() < config["diagnostic_analysis"]["min_zone_pixels"]:
            raise ValueError("laboratory zone too small")
        zones.append(ZoneMask(name, mask, 1))
    return tuple(zones)


def system_metrics():
    result = {"checked_monotonic_ns": time.monotonic_ns()}
    for label, command in (("throttled", ["vcgencmd", "get_throttled"]),
                           ("temperature", ["vcgencmd", "measure_temp"])):
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=2)
            result[label] = completed.stdout.strip() if completed.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            result[label] = "unavailable"
    return result


class LimitedJpegBuffer(BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, content):
        if self.tell() + len(content) > self.limit:
            raise ValueError("snapshot exceeds per-image budget")
        return super().write(content)


def run_observation(camera_factory, config, output, *, clock=time.monotonic_ns,
                    sleep=time.sleep, metrics=system_metrics, budget_factory=RunBudget):
    validate_config(config)
    budget = budget_factory(output, max_bytes=config["max_output_bytes"], reserve_bytes=config["reserve_bytes"])
    camera, samples, snapshots, error = None, 0, [], None
    started = False
    process_start = time.process_time_ns()
    try:
        camera = camera_factory()
        if "imx708" not in str(camera.camera_properties.get("Model", "")).lower():
            raise ValueError("this experiment requires an identified IMX708 camera")
        requested = camera.create_video_configuration(
            main={"size": tuple(config["main_size"]), "format": "YUV420"},
            lores={"size": tuple(config["lores_size"]), "format": "YUV420"},
            raw=None, display=None, queue=False, controls={"FrameRate": config["requested_fps"]},
            sensor={"output_size": tuple(config["sensor"]["output_size"]), "bit_depth": config["sensor"]["bit_depth"]},
        )
        requested_snapshot = json_safe(requested)
        camera.configure(requested)
        actual_config = camera.camera_configuration()
        for name, size in (("main", config["main_size"]), ("lores", config["lores_size"])):
            if (list(actual_config[name]["size"]) != size or actual_config[name]["format"] != "YUV420"):
                raise ValueError("actual stream differs from bounded configuration")
        if (list(actual_config["sensor"]["output_size"]) != config["sensor"]["output_size"]
                or actual_config["sensor"]["bit_depth"] != config["sensor"]["bit_depth"]):
            raise ValueError("actual sensor mode differs from requested mode")
        if "identity" not in str(actual_config.get("transform")):
            raise ValueError("actual transform is not identity")
        exp = camera.camera_controls["ExposureTime"]
        gain = camera.camera_controls["AnalogueGain"]
        hardware = ExposureLimits(int(exp[0]), int(exp[1]), float(gain[0]), float(gain[1]), 0, 0)
        advisor = ShadowAdvisor(hardware, policy_config(config))
        zones, analysis = laboratory_zones(config), AnalysisConfig(**config["diagnostic_analysis"])
        latest_metrics = metrics()
        setup = {
            "experiment_config": config, "requested_camera_configuration": requested_snapshot,
            "actual_camera_configuration": actual_config, "camera_controls": camera.camera_controls,
            "policy_hardware_limits": asdict(hardware), "system": latest_metrics,
            "available_storage_bytes": budget.available_bytes(), "python": platform.python_version(), "numpy": np.__version__,
            "camera_properties": camera.camera_properties, "versions": package_versions(),
            "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                              for name in ("road_pi_observe_v01.py", "road_observer_v01.py", "road_core_v02.py")},
            "exposure_controls_sent_by_policy": False, "sensor_hdr_state": "not explicitly set; not independently verified",
            "frame_id_kind": "SensorTimestamp surrogate, not sequential frame number",
        }
        budget.write_json("setup.json", setup)
        camera.start()
        started = True
        start = clock()
        last_metrics = start
        duration_ns = round(config["duration_s"] * 1e9)
        period_ns = round(config["sample_period_s"] * 1e9)
        max_samples = (duration_ns + period_ns - 1) // period_ns
        with budget.open_log() as log:
            while clock() - start < duration_ns and samples < max_samples:
                tick = clock()
                elapsed = (tick - start) / 1e9
                request = None
                encoded = None
                snapshot_index = next((i for i, at in enumerate(config["snapshot_at_s"])
                                       if elapsed >= at and i not in snapshots), None)
                job = camera.capture_request(wait=False, flush=True)
                try:
                    request = camera.wait(job, timeout=config["capture_timeout_s"])
                    received_ns = clock()
                    raw_md = request.get_metadata()
                    array = request.make_array("lores")
                    if snapshot_index is not None:
                        with LimitedJpegBuffer(config["snapshot_max_bytes"]) as buffer:
                            request.save("main", buffer, format="jpeg")
                            encoded = buffer.getvalue()
                finally:
                    if request is not None:
                        request.release()
                analysis_start = clock()
                now = analysis_start
                speed = (None if config["simulated_speed_kmh"] is None else
                         SpeedSample(config["simulated_speed_kmh"], now))
                try:
                    frame = frame_from_metadata(raw_md)
                    plane = visible_y_plane(array, actual_config["lores"])
                    scene = analyze_y_plane(plane, zones, analysis, mask_version="module3-lab-grid-001",
                                            transform="identity", calibrated=False)
                    record = advisor.observe(scene, frame, speed, now)
                except ValueError as invalid:
                    record = advisor.observe(None, None, speed, now)
                    record["adapter_input_error"] = str(invalid)
                analysis_end = clock()
                if now - last_metrics >= 5_000_000_000:
                    latest_metrics, last_metrics = metrics(), now
                metadata_dict = raw_md if isinstance(raw_md, dict) else {}
                record.update({
                    "sample_index": samples, "elapsed_ns": tick - start,
                    "capture_started_ns": tick, "capture_returned_ns": received_ns,
                    "analysis_ns": analysis_end - analysis_start,
                    "actual_metadata": {key: metadata_dict.get(key) for key in METADATA_FIELDS},
                    "unavailable_metadata": [key for key in METADATA_FIELDS if key not in metadata_dict],
                    "camera_controls_sent_this_sample": {}, "speed_source": "SIMULATED" if speed else "MISSING",
                    "system": latest_metrics, "available_storage_bytes": budget.available_bytes(),
                    "snapshot": None if encoded is None else f"frame_{samples:03d}.jpg",
                })
                if encoded is not None:
                    budget.write_new(record["snapshot"], encoded)
                    snapshots.append(snapshot_index)
                budget.append_record(log, record)
                samples += 1
                sleep(max(0, min(period_ns - (clock() - tick), duration_ns - (clock() - start))) / 1e9)
    except BaseException as failure:
        error = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        cleanup_error = None
        try:
            if camera is not None:
                try:
                    camera.stop()
                except BaseException as failure:
                    cleanup_error = "stop: " + type(failure).__name__ + ": " + str(failure)
                    raise
                finally:
                    try:
                        camera.close()
                    except BaseException as failure:
                        cleanup_error = "close: " + type(failure).__name__ + ": " + str(failure)
                        raise
        finally:
            summary = {"camera_started": started, "samples": samples, "snapshots": len(snapshots),
                       "command_dispatch_count": 0, "error": error, "system_after": metrics(),
                       "cleanup_error": cleanup_error,
                       "process_cpu_ns": time.process_time_ns() - process_start,
                       "bytes_before_summary": budget.used,
                       "limits": "Selected observations only; no encoded, transported or inference throughput measured."}
            try:
                budget.write_json("summary.json", summary)
            except (OSError, ValueError) as failure:
                print(f"Summary not saved: {failure}", file=sys.stderr)
    return summary


def package_versions():
    try:
        result = subprocess.run(["dpkg-query", "-W", "python3-picamera2", "python3-libcamera",
                                 "libcamera0.7", "rpicam-apps-core", "rpicam-apps-lite"],
                                capture_output=True, text=True, timeout=2)
        return {"dpkg_query_returncode": result.returncode, "packages": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired):
        return {"packages": "unavailable"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pi_observer_lab_v01.json")
    parser.add_argument("--run-camera", action="store_true")
    parser.add_argument("--allow-uncalibrated-lab", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if not args.run_camera:
            print(json.dumps({"dry_run": True, "camera_started": False, "config": config}, indent=2))
            return 0
        if not args.allow_uncalibrated_lab or args.output is None:
            raise ValueError("camera run requires explicit lab acknowledgement and a NEW --output directory")
        from picamera2 import Picamera2
        result = run_observation(lambda: Picamera2(0), config, args.output)
        print(json.dumps(json_safe(result)))
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, ImportError, KeyError) as failure:
        print(f"Observer failed: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
