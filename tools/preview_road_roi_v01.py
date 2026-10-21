"""Bounded local CVAT preview. No camera, network, source edits or full extraction.

Output directory must be NEW and inside the ignored runs/ directory. Original
images remain in their ZIP. Preview thumbnails are analysis artifacts only.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from road_core_v02 import AnalysisConfig, analyze_y_plane
from road_masks_v01 import build_metering_masks

COLORS = ((40, 200, 255), (70, 255, 100), (255, 130, 45))


def inspect_archive(path, config, profile, output):
    output = output.resolve()
    runs = (ROOT / "runs").resolve()
    if not output.is_relative_to(runs) or output == runs:
        raise ValueError("output must be a new directory inside project runs/")
    if output.exists():
        raise ValueError("output already exists; choose a new experiment directory")
    config_bytes = config.read_bytes()
    settings = json.loads(config_bytes)
    zones, exclude, review = build_metering_masks(settings, profile, (240, 320), transform="identity")
    analysis = AnalysisConfig(32, 235, 50, 16)  # JPEG diagnostics only; no exposure target.
    summaries, tiles = [], []
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 1000 or sum(entry.file_size for entry in entries) > 1_000_000_000:
            raise ValueError("archive exceeds inspection bounds")
        images = sorted((entry for entry in entries if entry.filename.lower().endswith((".jpg", ".jpeg"))),
                        key=lambda entry: entry.filename)
        if not images or len(images) > 300 or len({item.filename for item in images}) != len(images):
            raise ValueError("expected 1..300 uniquely named JPEGs")
        selected = set(int(value) for value in np.linspace(0, len(images) - 1, min(12, len(images))))
        for number, entry in enumerate(images):
            if not 0 < entry.file_size <= 5_000_000:
                raise ValueError("JPEG exceeds per-image bound")
            with Image.open(BytesIO(archive.read(entry))) as original:
                if original.size != (2592, 1944):
                    raise ValueError("this review configuration expects original 2592x1944 images")
                image = original.convert("RGB")
                small = image.resize((320, 240), Image.Resampling.BOX)
                # Encoded JPEG luminance diagnostic, not calibrated Picamera2 Y.
                scene = analyze_y_plane(np.asarray(small.convert("L")), zones, analysis,
                                        mask_version=settings["version"] + ":" + profile,
                                        transform="identity", exclude=exclude)
                summaries.append({"member": entry.filename, "statistics": asdict(scene)})
                if number in selected:
                    base = image.resize((640, 480), Image.Resampling.LANCZOS)
                    tint = np.asarray(base).copy()
                    for zone, color in zip(zones, COLORS):
                        active = np.asarray(Image.fromarray(zone.mask & ~exclude).resize(base.size, Image.Resampling.NEAREST))
                        tint[active] = (0.72 * tint[active] + 0.28 * np.asarray(color)).astype(np.uint8)
                    canvas = Image.fromarray(tint)
                    draw = ImageDraw.Draw(canvas)
                    # Explicit exclusions and review-only rectangle outlines.
                    for key, polygon in settings["exclusions"].items():
                        points = [(round(x * 639), round(y * 479)) for x, y in polygon]
                        draw.line(points + points[:1], fill=(255, 60, 60), width=2)
                    for polygon in settings["review_only"].values():
                        points = [(round(x * 639), round(y * 479)) for x, y in polygon]
                        draw.line(points + points[:1], fill=(255, 255, 0), width=2)
                    draw.rectangle((0, 0, 639, 39), fill=(0, 0, 0))
                    draw.text((6, 3), f"index={number}  {profile}", fill=(255, 255, 255))
                    draw.text((6, 19), "UNCALIBRATED / NO METRIC DISTANCE / COLOR=METERING ONLY", fill=(255, 255, 255))
                    tiles.append((number, canvas))
    # Finish reads before creating output; never reuse an existing experiment.
    output.mkdir(parents=True, exist_ok=False)
    with (output / "mask_config_001.json").open("xb") as stream:
        stream.write(config_bytes)
    sheet = Image.new("RGB", (640 * 3, 480 * 4), (30, 30, 30))
    for position, (number, tile) in enumerate(tiles):
        tile.save(output / f"preview_{number:03d}.png")
        sheet.paste(tile, ((position % 3) * 640, (position // 3) * 480))
    sheet.save(output / "contact_sheet.png")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    archive_sha = digest.hexdigest()
    result = {
        "schema_version": 1, "archive_sha256": archive_sha,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(), "profile": profile,
        "calibrated": False, "images_checked": len(summaries),
        "selected_sorted_indices": sorted(selected),
        "metering_pixels_320x240": {zone.name: int(np.sum(zone.mask & ~exclude)) for zone in zones},
        "diagnostic_thresholds": asdict(analysis),
        "note": "Sorted member order is not validated chronology. JPEG thumbnails, not sensor Y. No command policy is run.",
        "images": summaries,
    }
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Read {len(summaries)} images; saved {len(tiles)} previews and contact_sheet.png")
    print(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "road_roi_candidate_v01.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / ("roi_review_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    try:
        inspect_archive(args.archive, args.config, args.profile, output)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Preview failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
