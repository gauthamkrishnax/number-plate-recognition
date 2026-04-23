"""Run plate recognition on every image under test/ and save debug overlays to output/."""

import argparse
import sys
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
        "--pipeline",
        choices=["auto", "ml", "legacy"],
        default="auto",
        help="Same as app.py --pipeline.",
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
    for image_path in paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"skip (unreadable): {image_path.name}", file=sys.stderr)
            continue
        plate, bbox = app.detect_and_read_plate(image, pipeline=args.pipeline)
        if not plate:
            print(f"{image_path.name}\t(no plate)", file=sys.stderr)
            continue
        dest = out_dir / f"{image_path.stem}_result{image_path.suffix.lower()}"
        app.save_plate_debug_image(image, plate, bbox, dest)
        print(f"{image_path.name}\t{plate}\t-> {dest}")
        ok += 1

    print(f"Done: {ok}/{len(paths)} with a plate; debug images under {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
