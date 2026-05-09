"""
Faster plate detection than app.py defaults.

Strategy: run RapidOCR on a downscaled image with a small angle set and early
exit when a high-confidence plate-shaped hit is found; otherwise fall back to
``app.detect_and_read_plate`` (same accuracy as the original stack).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import app

# Longest image side (px) for the fast OCR passes. Smaller = faster, riskier.
_MAX_LONG_EDGE_FAST = 1280
# Strong enough to trust the fast pass without calling ``app`` (see ``app._score_ml_candidate``).
_GOOD_ENOUGH_SCORE = 162.0


def _resize_keep_aspect(
    image: np.ndarray, max_long_edge: int
) -> Tuple[np.ndarray, float]:
    """Returns (possibly resized image, scale from returned image to original)."""
    h, w = image.shape[:2]
    m = max(h, w)
    if m <= max_long_edge:
        return image, 1.0
    s = max_long_edge / m
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    out = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, w / float(nw)


def _scale_bbox_to_full(
    bbox: Tuple[int, int, int, int], inv_scale: float
) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    return (
        int(round(x * inv_scale)),
        int(round(y * inv_scale)),
        int(round(w * inv_scale)),
        int(round(h * inv_scale)),
    )


def _ml_eval_rotated(
    engine,
    rotated: np.ndarray,
    best_text: Optional[str],
    best_bbox: Optional[Tuple[int, int, int, int]],
    best_score: float,
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]], float]:
    """Same candidate logic as ``app.detect_and_read_plate_ml`` for one rotated frame."""
    result, _ = engine(rotated)
    if not result:
        return best_text, best_bbox, best_score

    ih, iw = rotated.shape[:2]
    entries = app._rapidocr_entries_for_plates(result)

    for cluster in app._cluster_ml_entries(entries):
        if len(cluster) > 6:
            continue
        idxs = sorted(cluster, key=lambda i: (entries[i]["y"], entries[i]["x"]))
        combined_raw = "".join(entries[i]["raw"] for i in idxs)
        if len(combined_raw) > 48:
            continue
        text = app.best_plate_substring(combined_raw)
        conf_mean = float(np.mean([entries[i]["conf"] for i in idxs]))
        xs = [entries[i]["x"] for i in idxs]
        ys = [entries[i]["y"] for i in idxs]
        x2s = [entries[i]["x2"] for i in idxs]
        y2s = [entries[i]["y2"] for i in idxs]
        x0, y0, x1, y1 = min(xs), min(ys), max(x2s), max(y2s)
        w0, h0 = x1 - x0, y1 - y0
        quality = app._score_ml_candidate(text, conf_mean, x0, y0, w0, h0, iw, ih)
        if quality > best_score:
            best_text = text
            best_bbox = (x0, y0, w0, h0)
            best_score = quality

    for entry in result:
        if len(entry) < 3:
            continue
        box_points, raw_text, raw_conf = entry[0], entry[1], entry[2]
        text = app.best_plate_substring(str(raw_text))
        if len(text) < 6:
            continue
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 0.0
        box = np.array(box_points, dtype=np.float32)
        x, y, w, h = cv2.boundingRect(box.astype(np.int32))
        if w < 28 or h < 10:
            continue
        el = app._box_elongation(w, h)
        if el < 1.25 or el > 12.0:
            continue
        quality = app._score_ml_candidate(text, confidence, x, y, w, h, iw, ih)
        if quality > best_score:
            best_text = text
            best_bbox = (x, y, w, h)
            best_score = quality

    return best_text, best_bbox, best_score


def _fast_pass_on_image(
    engine,
    work: np.ndarray,
    angles: List[float],
    inv_scale: float,
    per_angle_early_exit_score: float,
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]], float]:
    best_text: Optional[str] = None
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_score = -1.0

    for angle in angles:
        rotated = app.rotate_image(work, angle) if angle != 0 else work
        best_text, best_bbox, best_score = _ml_eval_rotated(
            engine, rotated, best_text, best_bbox, best_score
        )
        if best_score >= per_angle_early_exit_score and best_text and len(best_text) >= 8:
            break

    if best_bbox is not None and inv_scale > 1.0 + 1e-6:
        best_bbox = _scale_bbox_to_full(best_bbox, inv_scale)
    return best_text, best_bbox, best_score


def detect_and_read_plate_fast(
    image: np.ndarray,
    *,
    fallback: bool = True,
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
    text, bbox, _ = detect_and_read_plate_fast_with_score(image, fallback=fallback)
    return text, bbox


def detect_and_read_plate_fast_with_score(
    image: np.ndarray,
    *,
    fallback: bool = True,
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]], float]:
    """
    Prefer a few cheap RapidOCR passes; fall back to ``app`` when unsure or when
    RapidOCR is unavailable.
    """
    if app.get_rapidocr_engine() is None:
        return app.detect_and_read_plate_with_score(image, pipeline="auto")

    small, inv_scale = _resize_keep_aspect(image, _MAX_LONG_EDGE_FAST)

    engine = app.get_rapidocr_engine()
    assert engine is not None

    text, bbox, score = _fast_pass_on_image(
        engine,
        small,
        [0.0],
        inv_scale,
        _GOOD_ENOUGH_SCORE,
    )
    if text and score >= _GOOD_ENOUGH_SCORE:
        return text, bbox, score

    text2, bbox2, score2 = _fast_pass_on_image(
        engine,
        small,
        [-15.0, 15.0],
        inv_scale,
        _GOOD_ENOUGH_SCORE,
    )
    if score2 > score:
        text, bbox, score = text2, bbox2, score2
    if text and score >= _GOOD_ENOUGH_SCORE:
        return text, bbox, score

    if score < _GOOD_ENOUGH_SCORE:
        text3, bbox3, score3 = _fast_pass_on_image(
            engine,
            small,
            [-20.0, -10.0, -5.0, 5.0, 10.0, 20.0],
            inv_scale,
            _GOOD_ENOUGH_SCORE,
        )
        if score3 > score:
            text, bbox, score = text3, bbox3, score3
    if text and score >= _GOOD_ENOUGH_SCORE:
        return text, bbox, score

    if not fallback:
        return text, bbox, max(0.0, score)

    return app.detect_and_read_plate_with_score(image, pipeline="auto")


def process_image(image_path: Path, save_debug: bool, *, fallback: bool = True) -> int:
    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        return 2

    image = cv2.imread(str(image_path))
    if image is None:
        print("Error: could not read image file.")
        return 2

    extracted_plate, bbox = detect_and_read_plate_fast(image, fallback=fallback)
    if not extracted_plate:
        return 1

    print(extracted_plate)

    if save_debug:
        output_path = image_path.with_name(f"{image_path.stem}_result{image_path.suffix}")
        app.save_plate_debug_image(image, extracted_plate, bbox, output_path)
        print(f"Debug image saved to: {output_path}", file=sys.stderr)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recognize a plate using the faster app2 pipeline (fast ML pass, then app fallback)."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save an output image with detected plate region.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not call app.detect_and_read_plate if the fast pass is weak (for debugging).",
    )
    args = parser.parse_args()
    raise SystemExit(
        process_image(Path(args.image), args.save_debug, fallback=not args.no_fallback)
    )


if __name__ == "__main__":
    main()
