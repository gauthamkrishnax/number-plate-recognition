"""Run plate recognition on every image under test/ and save debug overlays to output/."""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    root = Path(__file__).resolve().parent
    venv_python = root / ".venv" / "bin" / "python"
    sys.stderr.write(
        "Missing OpenCV (cv2). Install project deps, then use the same interpreter:\n"
        f"  cd {root}\n"
        "  python3 -m venv .venv\n"
        "  .venv/bin/pip install -r requirements.txt\n"
    )
    if venv_python.is_file():
        sys.stderr.write(f"  {venv_python} {Path(__file__).name}\n")
    else:
        sys.stderr.write("  .venv/bin/python run_test_batch.py\n")
    raise SystemExit(1) from None

import app  # noqa: E402
import app2  # noqa: E402

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=root / "test",
        help="Directory of input images (default: <repo>/test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output",
        help="Directory for debug images (default: <repo>/output).",
    )
    parser.add_argument(
        "--backend",
        choices=["app", "app2"],
        default="app2",
        help="Recognition stack: app (app.py) or app2 (faster app2.py). Default: app2.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["auto", "ml", "legacy"],
        default="auto",
        help="Passed to app.py when --backend app (ignored for app2).",
    )
    args = parser.parse_args()

    test_dir: Path = args.test_dir
    out_dir: Path = args.output_dir

    if not test_dir.is_dir():
        print(f"Error: not a directory: {test_dir}", file=sys.stderr)
        raise SystemExit(2)

    paths = sorted(
        p
        for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not paths:
        print(f"No images found in {test_dir}", file=sys.stderr)
        raise SystemExit(1)

    ok = 0
    # Per-image detect+read timing (seconds, `perf_counter`).
    perf_rows: list[dict[str, object]] = []

    for image_path in paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"skip (unreadable): {image_path.name}", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        if args.backend == "app2":
            plate, bbox = app2.detect_and_read_plate_fast(image, fallback=True)
        else:
            plate, bbox = app.detect_and_read_plate(image, pipeline=args.pipeline)
        elapsed_s = time.perf_counter() - t0
        perf_rows.append(
            {
                "file": image_path.name,
                "detect_read_seconds": round(elapsed_s, 6),
                "plate_found": bool(plate),
            }
        )
        if not plate:
            print(f"{image_path.name}\t(no plate)", file=sys.stderr)
            continue
        dest = out_dir / f"{image_path.stem}_result{image_path.suffix.lower()}"
        app.save_plate_debug_image(image, plate, bbox, dest)
        print(f"{image_path.name}\t{plate}\t-> {dest}")
        ok += 1

    timed = [r["detect_read_seconds"] for r in perf_rows]
    timed_with_plate = [
        float(r["detect_read_seconds"]) for r in perf_rows if r["plate_found"]
    ]
    perf_summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_dir": str(test_dir),
        "output_dir": str(out_dir),
        "backend": args.backend,
        "pipeline": args.pipeline if args.backend == "app" else None,
        "image_files_total": len(paths),
        "images_timed": len(timed),
        "plates_found": ok,
        "avg_detect_read_seconds_all_timed": round(statistics.mean(timed), 6)
        if timed
        else None,
        "avg_detect_read_seconds_plate_found_only": round(
            statistics.mean(timed_with_plate), 6
        )
        if timed_with_plate
        else None,
        "min_seconds": round(min(timed), 6) if timed else None,
        "max_seconds": round(max(timed), 6) if timed else None,
        "per_image": perf_rows,
    }

    perf_path = out_dir / "batch_performance.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    perf_path.write_text(json.dumps(perf_summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"Done: {ok}/{len(paths)} with a plate; backend={args.backend}; "
        f"debug images under {out_dir}",
        file=sys.stderr,
    )
    if timed:
        print(
            f"Performance: avg detect+read over all {len(timed)} timed image(s): "
            f"{perf_summary['avg_detect_read_seconds_all_timed']} s",
            file=sys.stderr,
        )
    if timed_with_plate:
        print(
            f"Performance: avg over {len(timed_with_plate)} image(s) with a plate: "
            f"{perf_summary['avg_detect_read_seconds_plate_found_only']} s",
            file=sys.stderr,
        )
    print(f"Performance data written to {perf_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
