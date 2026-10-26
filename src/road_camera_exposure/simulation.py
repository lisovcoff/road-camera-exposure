"""Deterministic synthetic feedback example; NOT a calibrated camera simulator.

Uses linear uint8 brightness, fixed delay and quantized controls. No images are
read/written, no camera/network imports, no performance claims about real ISP.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .core import (AnalysisConfig, ExposureController, ExposureLimits,
                          FrameMetadata, PolicyConfig, Profile, SpeedSample,
                          ZoneMask, analyze_y_plane)

NS = 1_000_000_000


def synthetic_configuration():
    """Explicit test-only numbers, never defaults or recommended road settings."""
    hardware = ExposureLimits(100, 20_000, 1.0, 16.0, 25, 0.05)
    config = PolicyConfig(
        version="synthetic-only-001",
        detect=Profile(100, 4000, 4000, 100, 0.15),
        view=Profile(100, 12_000, 8000, 105, 0.15),
        enter_detect_kmh=9, enter_view_kmh=11,
        mode_dwell_ns=2 * NS, status_dwell_ns=NS // 5,
        max_speed_age_ns=NS // 2, max_frame_age_ns=NS // 2,
        command_timeout_ns=NS, min_response=0.8, max_response=1.25,
        highlight_pct=1.0,
    )
    return hardware, config, AnalysisConfig(32, 235, 50, 4)


def run_synthetic():
    hardware, config, analysis = synthetic_configuration()
    controller = ExposureController(hardware, config)
    near = np.zeros((20, 20), dtype=bool)
    near[10:, :] = True
    zones = (ZoneMask("near", near, 1), ZoneMask("far", ~near, 1))
    exposure, gain = 4000, 8.0
    scheduled = None
    rows = []
    for frame in range(60):
        now = NS + frame * (NS // 10)
        if scheduled is not None and frame >= scheduled[0]:
            command = scheduled[1]
            # Simple quantization only; no actual sensor model.
            exposure = max(100, round(command.exposure_us / 20) * 20)
            gain = round(command.analogue_gain * 32) / 32
            scheduled = None
        phase = "DIM" if frame < 15 or frame >= 40 else "BRIGHT"
        full_scale_y = 22 if phase == "DIM" else 160
        y = int(np.clip(round(full_scale_y * exposure * gain / 64_000), 0, 255))
        scene = analyze_y_plane(np.full((20, 20), y, dtype=np.uint8), zones, analysis,
                                mask_version="synthetic-grid-001", transform="identity")
        metadata = FrameMetadata(frame, now - 10_000_000, exposure, gain, 1.0, 33333)
        decision = controller.update(scene, metadata, SpeedSample(8, now), now)
        row = decision.to_record()
        row["synthetic_lighting"] = phase
        rows.append(row)
        if decision.command:
            if scheduled is not None:
                raise AssertionError("controller accumulated unobserved commands")
            controller.mark_sent(decision.command.command_id, now)
            scheduled = (frame + 2, decision.command)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="JSONL telemetry to stdout only")
    args = parser.parse_args(argv)
    rows = run_synthetic()
    if args.json:
        for row in rows:
            print(json.dumps(row, allow_nan=False))
    else:
        print("SYNTHETIC ONLY: linear brightness, 2-step delay, quantization; not road validation.")
        previous = None
        for row in rows:
            state = (row["synthetic_lighting"], tuple(row["statuses"]))
            if state != previous:
                t = (row["now_ns"] - NS) / NS
                print(f"t={t:.1f}s {state[0]} Y={row['statistics']['meter_y']:.0f} "
                      f"{row['actual']['exposure_us']}us/{row['actual']['analogue_gain']:.3f}x "
                      + ",".join(row["statuses"]))
                previous = state
        print(f"frames={len(rows)} commands={sum(row['command'] is not None for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
