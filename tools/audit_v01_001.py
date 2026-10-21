"""Read-only replay of recorded v0.1 feedback. Never imports a camera adapter.

Recorded measurements are inputs, not a simulation of how new controls would
change an image. Timestamp sensitivity is diagnostic and never repairs a log.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from road_core_v01 import ExposureController, RoiStats  # noqa: E402


def read_records(path: Path) -> list[dict]:
    """Reject malformed feedback before passing it to the unvalidated v0.1 core."""
    records = []
    previous = -math.inf
    required = ("elapsed_s", "simulated_speed_kmh", "analysis_ms")
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            values = [row[key] for key in required]
            values += [row["stats"][key] for key in RoiStats.__dataclass_fields__]
            values += [row["actual_metadata"][key]
                       for key in ("ExposureTime", "AnalogueGain")]
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in values):
                raise ValueError("feedback must contain finite numbers")
            if row["elapsed_s"] < 0 or row["elapsed_s"] <= previous:
                raise ValueError("elapsed_s must be nonnegative and strictly increasing")
            if row["simulated_speed_kmh"] < 0 or row["analysis_ms"] < 0:
                raise ValueError("speed and analysis_ms must be nonnegative")
            if row["actual_metadata"]["ExposureTime"] <= 0 or row["actual_metadata"]["AnalogueGain"] <= 0:
                raise ValueError("actual exposure and gain must be positive")
            if not isinstance(row["decision"], dict):
                raise ValueError("decision must be an object")
            previous = row["elapsed_s"]
            records.append(row)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path.name}:{number}: {error}") from error
    if not records:
        raise ValueError(f"{path.name}: no records")
    return records


def replay(records: list[dict], *, boundary_probe: bool = False) -> list[dict]:
    controller = ExposureController()
    mismatches = []
    for number, row in enumerate(records, 1):
        elapsed = row["elapsed_s"]
        # Probe a narrow numerical boundary only. This is NOT recovered timing.
        if (boundary_probe and controller.pending_since is not None
                and abs((elapsed - controller.pending_since) - 2.0) <= 1e-9):
            elapsed += 1e-6
        metadata = row["actual_metadata"]
        calculated = controller.decide(
            RoiStats(**row["stats"]), metadata["ExposureTime"],
            metadata["AnalogueGain"], row["simulated_speed_kmh"], elapsed,
        )
        if calculated != row["decision"]:
            mismatches.append({"line": number, "elapsed_s": row["elapsed_s"],
                               "recorded": row["decision"], "replayed": calculated})
    return mismatches


def audit(path: Path) -> dict:
    records = read_records(path)
    mismatches = replay(records)
    sensitive = replay(records, boundary_probe=True)
    actual = [row["actual_metadata"] for row in records]
    times = [row["analysis_ms"] for row in records]
    return {
        "dataset": path.parent.name if path.name == "measurements.jsonl" else path.stem,
        "rows": len(records),
        "exact_replay_matches": len(records) - len(mismatches),
        "exact_replay_mismatches": mismatches,
        "boundary_probe_mismatch_count": len(sensitive),
        "boundary_probe_note": "Adds 1 us only at a 2 s boundary within 1 ns; diagnostic, not proof of original timing.",
        "actual_exposure_us_range": [min(x["ExposureTime"] for x in actual), max(x["ExposureTime"] for x in actual)],
        "actual_gain_range": [min(x["AnalogueGain"] for x in actual), max(x["AnalogueGain"] for x in actual)],
        "analysis_ms_median": statistics.median(times),
        "analysis_ms_max": max(times),
        "limits": ["Recorded stationary-scene feedback, not a closed-loop simulation.",
                   "Does not establish encoded, delivered, or inference frame rate.",
                   "Cannot establish actual lamp-switch time or road detection quality."],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path,
                        help="Optional local directory containing experiment/measurements.jsonl")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout; no files are written")
    args = parser.parse_args(argv)
    paths = ([args.examples / name / "measurements.jsonl"
              for name in ("auto_controller", "light_response")] if args.examples else
             [ROOT / "tests" / "fixtures" / "v01" / f"{name}.jsonl"
              for name in ("auto_controller", "light_response")])
    try:
        reports = [audit(path) for path in paths]
    except (OSError, ValueError) as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(reports, indent=2, allow_nan=False))
    else:
        for report in reports:
            print(f"{report['dataset']}: {report['rows']} rows; exact replay "
                  f"{report['exact_replay_matches']}/{report['rows']}; "
                  f"boundary-probe mismatches={report['boundary_probe_mismatch_count']}")
            print(f"  analysis median/max: {report['analysis_ms_median']:.4f}/"
                  f"{report['analysis_ms_max']:.3f} ms")
            for mismatch in report["exact_replay_mismatches"]:
                print(f"  MISMATCH line {mismatch['line']} at {mismatch['elapsed_s']} s")
        print("Recorded-feedback audit only; not a hardware or road-performance test.")
    # Any exact mismatch remains visible to automation, including the known one.
    return 1 if any(report["exact_replay_mismatches"] for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
