import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None

PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$"),  # e.g. KA53MK2655
    re.compile(r"^\d{2}[A-Z]{2}\d{4}[A-Z]?$"),      # e.g. 22BH6517A
]
_RAPID_OCR_ENGINE = None
# Favor larger plate regions when several plausible readings exist (e.g. background plates).
_PLATE_AREA_SCORE_WEIGHT = 220.0
# Favor plates nearer the image center (0 = off, higher = stronger center bias).
_PLATE_CENTER_SCORE_WEIGHT = 65.0


def normalize_plate(text: str) -> str:
    """Keep only letters and numbers for robust matching."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def plate_pattern_bonus(text: str) -> float:
    if not text:
        return -20.0
    for pattern in PLATE_PATTERNS:
        if pattern.match(text):
            return 30.0
    if re.search(r"\d{2}[A-Z]{2}\d{4}", text):
        return 12.0
    return -5.0


def best_plate_substring(text: str) -> str:
    normalized = normalize_plate(text)
    if len(normalized) < 6:
        return ""

    windows = []
    for size in range(7, min(len(normalized), 11) + 1):
        for start in range(0, len(normalized) - size + 1):
            windows.append(normalized[start : start + size])

    best = ""
    best_score = -1000.0
    for candidate in windows:
        length_bonus = 10.0 if 8 <= len(candidate) <= 10 else 0.0
        score = plate_pattern_bonus(candidate) + len(candidate) * 2.0 + length_bonus
        if score > best_score:
            best = candidate
            best_score = score
    return best if best else ""


def get_rapidocr_engine():
    global _RAPID_OCR_ENGINE
    if RapidOCR is None:
        return None
    if _RAPID_OCR_ENGINE is None:
        _RAPID_OCR_ENGINE = RapidOCR()
    return _RAPID_OCR_ENGINE


def _box_elongation(w: int, h: int) -> float:
    """>= 1; high for long thin boxes (car plates) and tall stacked bike plates."""
    return max(w / float(max(h, 1)), h / float(max(w, 1)))


def _ml_stack_compatible(a: dict, b: dict) -> bool:
    """True if two OCR boxes likely belong to one two-line (bike) plate."""
    if a["y"] <= b["y"]:
        top, bot = a, b
    else:
        top, bot = b, a
    if abs(top["cx"] - bot["cx"]) > 0.42 * max(top["w"], bot["w"]):
        return False
    gap = bot["y"] - top["y2"]
    if gap > 1.05 * max(top["h"], bot["h"]):
        return False
    if gap < -0.32 * min(top["h"], bot["h"]):
        return False
    x_ov = min(top["x2"], bot["x2"]) - max(top["x"], bot["x"])
    if x_ov < 0.22 * min(top["w"], bot["w"]):
        return False
    return True


def _rapidocr_entries_for_plates(result) -> List[dict]:
    """Parse RapidOCR lines into axis-aligned boxes with metadata."""
    entries: List[dict] = []
    for entry in result:
        if len(entry) < 3:
            continue
        box_points, raw_text, raw_conf = entry[0], entry[1], entry[2]
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 0.0
        box = np.array(box_points, dtype=np.float32)
        x, y, w, h = cv2.boundingRect(box.astype(np.int32))
        if w < 28 or h < 10:
            continue
        el = _box_elongation(w, h)
        if el < 1.2 or el > 12.0:
            continue
        entries.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "x2": x + w,
                "y2": y + h,
                "cx": x + w * 0.5,
                "cy": y + h * 0.5,
                "raw": str(raw_text),
                "conf": confidence,
            }
        )
    return entries


def _cluster_ml_entries(entries: List[dict]) -> List[List[int]]:
    """Group vertically stacked boxes (typical two-wheeler HSRP) via union-find."""
    n = len(entries)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            if _ml_stack_compatible(entries[i], entries[j]):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def _bbox_area_fraction(w: int, h: int, iw: int, ih: int) -> float:
    return (max(w, 1) * max(h, 1)) / float(max(iw * ih, 1))


def _bbox_center_closeness(x: int, y: int, w: int, h: int, iw: int, ih: int) -> float:
    """1.0 when bbox center is at image center; ~0.0 toward frame corners (linear falloff)."""
    if iw <= 0 or ih <= 0:
        return 0.0
    cx = x + w * 0.5
    cy = y + h * 0.5
    icx = iw * 0.5
    icy = ih * 0.5
    dist = float(np.hypot(cx - icx, cy - icy))
    max_dist = float(np.hypot(iw * 0.5, ih * 0.5))
    if max_dist <= 1e-6:
        return 1.0
    return max(0.0, 1.0 - min(1.0, dist / max_dist))


def _score_ml_candidate(
    text: str,
    confidence: float,
    x: int,
    y: int,
    w: int,
    h: int,
    iw: int,
    ih: int,
) -> float:
    if len(text) < 6:
        return -1.0
    area_bonus = _PLATE_AREA_SCORE_WEIGHT * _bbox_area_fraction(w, h, iw, ih)
    center_bonus = _PLATE_CENTER_SCORE_WEIGHT * _bbox_center_closeness(x, y, w, h, iw, ih)
    return (
        confidence * 40.0
        + plate_pattern_bonus(text) * 2.0
        + len(text) * 4.0
        + area_bonus
        + center_bonus
    )


def detect_and_read_plate_ml(image) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
    """
    PP-OCR based detector+recognizer pipeline.
    This is a true ML stage that detects text boxes and recognizes their content.
    Supports single-line car plates and stacked two-line bike plates.
    """
    engine = get_rapidocr_engine()
    if engine is None:
        return None, None

    candidate_angles = [0, -20, -10, -5, 5, 10, 20]
    best_text: Optional[str] = None
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_score = -1.0

    for angle in candidate_angles:
        rotated = rotate_image(image, angle) if angle != 0 else image
        ih, iw = rotated.shape[:2]
        result, _ = engine(rotated)
        if not result:
            continue

        entries = _rapidocr_entries_for_plates(result)

        for cluster in _cluster_ml_entries(entries):
            if len(cluster) > 6:
                continue
            idxs = sorted(cluster, key=lambda i: (entries[i]["y"], entries[i]["x"]))
            combined_raw = "".join(entries[i]["raw"] for i in idxs)
            if len(combined_raw) > 48:
                continue
            text = best_plate_substring(combined_raw)
            conf_mean = float(np.mean([entries[i]["conf"] for i in idxs]))
            xs = [entries[i]["x"] for i in idxs]
            ys = [entries[i]["y"] for i in idxs]
            x2s = [entries[i]["x2"] for i in idxs]
            y2s = [entries[i]["y2"] for i in idxs]
            x0, y0, x1, y1 = min(xs), min(ys), max(x2s), max(y2s)
            w0, h0 = x1 - x0, y1 - y0
            quality = _score_ml_candidate(text, conf_mean, x0, y0, w0, h0, iw, ih)
            if quality > best_score:
                best_text = text
                best_bbox = (x0, y0, w0, h0)
                best_score = quality

        for entry in result:
            if len(entry) < 3:
                continue
            box_points, raw_text, raw_conf = entry[0], entry[1], entry[2]
            text = best_plate_substring(str(raw_text))
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
            el = _box_elongation(w, h)
            if el < 1.25 or el > 12.0:
                continue

            quality = _score_ml_candidate(text, confidence, x, y, w, h, iw, ih)
            if quality > best_score:
                best_text = text
                best_bbox = (x, y, w, h)
                best_score = quality

    return best_text, best_bbox


def order_box_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]  # top-left
    rect[2] = points[np.argmax(sums)]  # bottom-right
    diffs = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diffs)]  # top-right
    rect[3] = points[np.argmax(diffs)]  # bottom-left
    return rect


def warp_plate_from_box(image, box_points: np.ndarray):
    rect = order_box_points(box_points.astype("float32"))
    (tl, tr, br, bl) = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    if max_width < 60 or max_height < 15:
        return None, None

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    x, y, w, h = cv2.boundingRect(box_points.astype(np.int32))
    return warped, (x, y, w, h)


def detect_plate_regions(image) -> List[Tuple[any, Tuple[int, int, int, int]]]:
    """
    Detect possible plate regions with perspective correction.
    Returns candidate plate crops with axis-aligned bboxes for debug drawing.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]
    candidates: List[Tuple[any, Tuple[int, int, int, int]]] = []

    for contour in contours:
        if cv2.contourArea(contour) < 700:
            continue

        rect = cv2.minAreaRect(contour)
        (rw, rh) = rect[1]
        if rw <= 0 or rh <= 0:
            continue

        short_side = min(rw, rh)
        long_side = max(rw, rh)
        aspect_ratio = long_side / short_side if short_side > 0 else 0
        # Include taller two-wheeler plate outlines, not only wide car plates.
        if not (2.0 <= aspect_ratio <= 10.0 and short_side >= 15 and long_side >= 60):
            continue

        box = cv2.boxPoints(rect)
        plate, bbox = warp_plate_from_box(image, box)
        if plate is None or bbox is None:
            continue

        candidates.append((plate, bbox))
        if len(candidates) >= 8:
            break

    return candidates


def rotate_image(image, angle: float):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _ocr_best_text(binary_images, psm_values: List[int]) -> Tuple[str, float]:
    best_text = ""
    best_score = -1.0

    for binary in binary_images:
        for psm in psm_values:
            config = (
                f"--oem 3 --psm {psm} "
                "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
            data = pytesseract.image_to_data(
                binary, config=config, output_type=pytesseract.Output.DICT
            )
            cleaned = best_plate_substring(" ".join(data.get("text", [])))
            if len(cleaned) < 6:
                continue

            confidences = []
            for conf in data.get("conf", []):
                try:
                    value = float(conf)
                except (TypeError, ValueError):
                    continue
                if value >= 0:
                    confidences.append(value)
            confidence_score = float(np.mean(confidences)) if confidences else -1.0
            quality = confidence_score + len(cleaned) * 4.0 + plate_pattern_bonus(cleaned)
            if quality > best_score:
                best_text = cleaned
                best_score = quality

    return best_text, best_score


def extract_text_from_plate(plate_region) -> Tuple[str, float]:
    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 45, 45)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    binary_images = [thresh, morph, cv2.bitwise_not(thresh), adaptive]
    return _ocr_best_text(binary_images, [7, 8, 13])


def extract_text_from_image_fallback(image) -> Tuple[Optional[str], float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 35, 35)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 7
    )
    morph = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    best_text, best_score = _ocr_best_text(
        [thresh, morph, cv2.bitwise_not(thresh), adaptive], [6, 11, 13]
    )
    return (best_text or None), best_score


def detect_and_read_plate_legacy(image) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
    # Try several small rotations around the original angle to handle tilted photos.
    candidate_angles = [0, -20, -15, -10, -5, 5, 10, 15, 20]
    best_text: Optional[str] = None
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_score = -1.0

    for angle in candidate_angles:
        rotated = rotate_image(image, angle) if angle != 0 else image
        ih, iw = rotated.shape[:2]
        regions = detect_plate_regions(rotated)
        for plate_region, bbox in regions:
            text, score = extract_text_from_plate(plate_region)
            if not text:
                continue

            x, y, w, h = bbox
            area_bonus = _PLATE_AREA_SCORE_WEIGHT * _bbox_area_fraction(w, h, iw, ih)
            center_bonus = _PLATE_CENTER_SCORE_WEIGHT * _bbox_center_closeness(
                x, y, w, h, iw, ih
            )
            quality = score + len(text) * 5.0 + area_bonus + center_bonus
            if quality > best_score:
                best_text = text
                best_bbox = bbox
                best_score = quality

        # Fallback OCR on full rotated image for heavily skewed/blurred cases.
        fallback_text, fallback_score = extract_text_from_image_fallback(rotated)
        if fallback_text:
            # No localized bbox: omit area bonus so tight crops are preferred when scores are close.
            fallback_quality = fallback_score + len(fallback_text) * 3.0
            if fallback_quality > best_score:
                best_text = fallback_text
                best_bbox = None
                best_score = fallback_quality

    return best_text, best_bbox


def detect_and_read_plate(
    image, pipeline: str = "auto"
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
    if pipeline not in {"auto", "ml", "legacy"}:
        raise ValueError("pipeline must be one of: auto, ml, legacy")

    if pipeline == "legacy":
        return detect_and_read_plate_legacy(image)

    if pipeline in {"auto", "ml"}:
        ml_text, ml_bbox = detect_and_read_plate_ml(image)
        if ml_text:
            return ml_text, ml_bbox
        if pipeline == "ml":
            return None, None

    return detect_and_read_plate_legacy(image)


def save_plate_debug_image(
    image,
    plate: str,
    bbox: Optional[Tuple[int, int, int, int]],
    output_path: Path,
) -> None:
    """Write a copy of the image with the plate string and optional bounding box."""
    debug = image.copy()
    color = (0, 255, 0)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            debug,
            plate,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            debug,
            plate,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug)


def process_image(image_path: Path, save_debug: bool, pipeline: str) -> int:
    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        return 2

    image = cv2.imread(str(image_path))
    if image is None:
        print("Error: could not read image file.")
        return 2

    extracted_plate, bbox = detect_and_read_plate(image, pipeline=pipeline)
    if not extracted_plate:
        return 1

    print(extracted_plate)

    if save_debug:
        output_path = image_path.with_name(f"{image_path.stem}_result{image_path.suffix}")
        save_plate_debug_image(image, extracted_plate, bbox, output_path)
        print(f"Debug image saved to: {output_path}", file=sys.stderr)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recognize a vehicle number plate and print the extracted text."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save an output image with detected plate region.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["auto", "ml", "legacy"],
        default="auto",
        help="Recognition pipeline: auto (ML then fallback), ml (PP-OCR only), legacy (contour + Tesseract).",
    )
    args = parser.parse_args()

    exit_code = process_image(Path(args.image), args.save_debug, args.pipeline)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
