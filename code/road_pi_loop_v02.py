"""Bounded closed-loop exposure/gain test in operating gain range 2..16, no encoder/network.

Dry-run by default. Simulated speed and lab ROI are not road calibration.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

from road_core_v02 import ExposureController, ExposureLimits, SpeedSample
from road_observer_v01 import RunBudget, json_safe
from road_pi_controls_v01 import capture, configure_lab, setup_record
from road_pi_observe_v01 import ROOT, load_config, policy_config, package_versions, system_metrics


def verified_limits(calibration, camera, actual):
    if (calibration.get("schema_version") != 1 or calibration.get("accepted") is not True
            or calibration.get("camera_model") != str(camera.camera_properties["Model"])
            or calibration.get("actual_camera_configuration") != json_safe(actual)
            or calibration.get("versions") != package_versions()):
        raise ValueError("accepted calibration for identical camera/mode/software required")
    tolerance, gain_tolerance = calibration.get("exposure_tolerance_us"), calibration.get("gain_tolerance")
    if type(tolerance) is not int or not 0 <= tolerance <= 82 or type(gain_tolerance) not in (int, float) or not 0 <= gain_tolerance <= .31:
        raise ValueError("calibration tolerances outside bounded experiment")
    # Enforced operating subset, NOT the sensor's full advertised physical range.
    return ExposureLimits(100, 12000, 2., 16., tolerance, gain_tolerance)


def simulated_speed(elapsed):
    return 8. if elapsed < 5 or elapsed >= 10 else 13.


def run(camera_factory, calibration_path, output, clock=time.monotonic_ns):
    calibration_bytes = Path(calibration_path).read_bytes()
    calibration = json.loads(calibration_bytes)
    if calibration.get("accepted") is not True:
        raise ValueError("rejected calibration")
    budget = RunBudget(output, max_bytes=5_000_000, reserve_bytes=1_000_000_000)
    camera, error, cleanup, samples, sent_count, observed_count = None, None, [], 0, 0, 0
    cpu = time.process_time_ns()
    last = None
    try:
        camera = camera_factory()
        config = load_config(ROOT / "configs/pi_observer_lab_v01.json")
        requested, actual = configure_lab(camera)
        limits = verified_limits(calibration, camera, actual)
        policy = policy_config(config)
        policy = replace(policy, version="pi-loop-exposure-gain-002-NOT-road-calibrated",
                         view=replace(policy.view, max_exposure_us=8000),
                         command_timeout_ns=1_000_000_000)
        controller = ExposureController(limits, policy)
        setup = setup_record(camera, requested, actual)
        setup.update({"calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
                      "policy": asdict(policy), "operating_limits": asdict(limits),
                      "duration_s": 16, "max_samples": 520,
                      "gain_test_scope": "ADAPTIVE 2..16 with empirical readback envelope",
                      "speed_source": "SIMULATED 8/13/8 at 0/5/10 seconds",
                      "source_loop_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        budget.write_json("setup.json", setup)
        camera.set_controls({"AeEnable": False, "ExposureTime": 2000, "AnalogueGain": 2.})
        camera.start()
        for _ in range(18):
            capture(camera, actual, config, clock)
        start = clock()
        with budget.open_log() as log:
            while clock() - start < 16_000_000_000 and samples < 520:
                frame, scene, record = capture(camera, actual, config, clock)
                now = clock()
                decision = controller.update(scene, frame, SpeedSample(simulated_speed((now-start)/1e9), now), now)
                record.update({"sample_index": samples, "elapsed_ns": now-start,
                               "decision": decision.to_record(), "command_dispatch": None,
                               "speed_source": "SIMULATED", "gain_scope": "ADAPTIVE_2_TO_16"})
                if decision.observed_command_id is not None:
                    observed_count += 1
                if decision.command is not None:
                    command = decision.command
                    if not 2 <= command.analogue_gain <= 16 or not 100 <= command.exposure_us <= 8000:
                        raise ValueError("policy command outside bounded experiment")
                    sent = clock()
                    camera.set_controls({"AeEnable": False, "ExposureTime": command.exposure_us,
                                         "AnalogueGain": command.analogue_gain})
                    returned = clock()
                    controller.mark_sent(command.command_id, sent)
                    record["command_dispatch"] = {"command_id": command.command_id,
                        "started_ns": sent, "returned_ns": returned,
                        "requested_exposure_us": command.exposure_us, "requested_gain": command.analogue_gain}
                    sent_count += 1
                budget.append_record(log, record)
                last = decision
                samples += 1
                if any(s in decision.statuses for s in ("CONTROL_TIMEOUT", "INVALID_INPUT", "STALE_FRAME")):
                    raise RuntimeError("stopping control loop: " + str(decision.statuses))
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
        summary = {"samples": samples, "commands_sent": sent_count, "commands_observed": observed_count,
                   "pending_command_id": None if last is None else last.pending_command_id,
                   "error": error, "cleanup_errors": cleanup, "process_cpu_ns": time.process_time_ns()-cpu,
                   "system_after": system_metrics(), "available_storage_bytes": budget.available_bytes(),
                   "scope": "exposure and gain loop, simulated speed, uncalibrated lab ROI"}
        budget.write_json("summary.json", summary)
        if cleanup and error is None:
            raise RuntimeError("cleanup failed")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-camera", action="store_true")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.run_camera:
        print(json.dumps({"dry_run": True, "camera_started": False, "duration_s":16,
                          "gain_range":[2,16], "exposure_ceiling_us":[4000,8000], "speed":"SIMULATED"}))
        return 0
    if args.output is None or args.calibration is None:
        parser.error("NEW --output and accepted --calibration are required")
    from picamera2 import Picamera2
    print(json.dumps(json_safe(run(lambda: Picamera2(0), args.calibration, args.output))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
