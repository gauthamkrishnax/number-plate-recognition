import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

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
AMBIGUOUS_SUBS = {
    "0": ["O", "D", "Q"],
    "1": ["I", "L", "T"],
    "2": ["Z"],
    "5": ["S"],
    "6": ["G"],
    "8": ["B"],
    "A": ["4"],
    "B": ["8"],
    "D": ["0"],
    "G": ["6"],
    "I": ["1", "L"],
    "L": ["1", "I"],
    "O": ["0", "Q"],
    "Q": ["0", "O"],
    "S": ["5"],
    "T": ["1"],
    "Z": ["2"],
}
_RAPID_OCR_ENGINE = None


def normalize_plate(text: str) -> str:
    """Keep only letters and numbers for robust matching."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def load_allowlist(file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Allowlist file not found: {file_path}")

    entries: List[str] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        cleaned = normalize_plate(line.strip())
        if cleaned:
            entries.append(cleaned)
    return entries


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


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            repl = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, repl))
        prev = curr
    return prev[-1]


def corrected_distance(a: str, b: str) -> int:
    if not a and not b:
        return 0
    max_len = max(len(a), len(b))
    a = a.ljust(max_len, "_")
    b = b.ljust(max_len, "_")
    penalty = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            continue
        if cb in AMBIGUOUS_SUBS.get(ca, []) or ca in AMBIGUOUS_SUBS.get(cb, []):
            penalty += 0
        else:
            penalty += 1
    return penalty + abs(len(a.strip("_")) - len(b.strip("_")))


def fuzzy_allowlist_match(text: str, allowlist: List[str]) -> Optional[str]:
    if not text or not allowlist:
        return None
    best_match = None
    best_score = 999
    for plate in allowlist:
        d = min(levenshtein_distance(text, plate), corrected_distance(text, plate))
        if d < best_score:
            best_match = plate
            best_score = d
    if best_match is not None and best_score <= 2:
        return best_match
    return None


def get_rapidocr_engine():
    global _RAPID_OCR_ENGINE
    if RapidOCR is None:
        return None
    if _RAPID_OCR_ENGINE is None:
        _RAPID_OCR_ENGINE = RapidOCR()
    return _RAPID_OCR_ENGINE


def detect_and_read_plate_ml(image) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
    """
    PP-OCR based detector+recognizer pipeline.
    This is a true ML stage that detects text boxes and recognizes their content.
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
        result, _ = engine(rotated)
        if not result:
            continue

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
            if w < 45 or h < 12:
                continue
            aspect_ratio = w / float(max(h, 1))
            if not 1.6 <= aspect_ratio <= 9.0:
                continue

            quality = (
                confidence * 40.0
                + plate_pattern_bonus(text) * 2.0
                + len(text) * 4.0
            )
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
        if not (2.0 <= aspect_ratio <= 7.5 and short_side >= 15 and long_side >= 60):
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
        regions = detect_plate_regions(rotated)
        for plate_region, bbox in regions:
            text, score = extract_text_from_plate(plate_region)
            if not text:
                continue

            # Prefer longer confident plate strings.
            quality = score + len(text) * 5.0
            if quality > best_score:
                best_text = text
                best_bbox = bbox
                best_score = quality

        # Fallback OCR on full rotated image for heavily skewed/blurred cases.
        fallback_text, fallback_score = extract_text_from_image_fallback(rotated)
        if fallback_text:
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


def process_image(image_path: Path, allowlist_path: Path, save_debug: bool, pipeline: str) -> int:
    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        return 2

    allowlist = load_allowlist(allowlist_path)
    allowset = set(allowlist)

    image = cv2.imread(str(image_path))
    if image is None:
        print("Error: could not read image file.")
        return 2

    extracted_plate, bbox = detect_and_read_plate(image, pipeline=pipeline)
    if not extracted_plate:
        print("No plate detected.")
        return 1

    matched_plate = extracted_plate if extracted_plate in allowset else fuzzy_allowlist_match(
        extracted_plate, allowlist
    )
    is_match = matched_plate is not None
    status = "SUCCESS" if is_match else "FAIL"

    print(f"Detected plate: {extracted_plate}")
    if matched_plate and matched_plate != extracted_plate:
        print(f"Matched allowlist as: {matched_plate}")
    print(f"Match status: {status}")

    if save_debug and bbox is not None:
        x, y, w, h = bbox
        debug = image.copy()
        color = (0, 255, 0) if is_match else (0, 0, 255)
        cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            debug,
            f"{extracted_plate} ({status})",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        output_path = image_path.with_name(f"{image_path.stem}_result{image_path.suffix}")
        cv2.imwrite(str(output_path), debug)
        print(f"Debug image saved to: {output_path}")

    return 0 if is_match else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recognize a vehicle number plate and verify against an allowlist."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--allowlist",
        default="allowed_plates.txt",
        help="Path to file containing allowed plate values (one per line).",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save an output image with detected plate and match status.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["auto", "ml", "legacy"],
        default="auto",
        help="Recognition pipeline: auto (ML then fallback), ml (PP-OCR only), legacy (contour + Tesseract).",
    )
    args = parser.parse_args()

    exit_code = process_image(
        Path(args.image), Path(args.allowlist), args.save_debug, args.pipeline
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
