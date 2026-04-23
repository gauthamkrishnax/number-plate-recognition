import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import pytesseract


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


def detect_plate_region(image) -> Tuple[Optional[any], Optional[Tuple[int, int, int, int]]]:
    """
    Detect a possible number plate region using contour approximation.
    Returns cropped region and bounding box if found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        # A number plate is often close to a 4-corner polygon.
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2.0 <= aspect_ratio <= 6.5 and w > 60 and h > 15:
                plate = image[y : y + h, x : x + w]
                return plate, (x, y, w, h)

    return None, None


def extract_text_from_plate(plate_region) -> str:
    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    raw_text = pytesseract.image_to_string(thresh, config=config)
    return normalize_plate(raw_text)


def process_image(image_path: Path, allowlist_path: Path, save_debug: bool) -> int:
    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        return 2

    allowlist = load_allowlist(allowlist_path)
    allowset = set(allowlist)

    image = cv2.imread(str(image_path))
    if image is None:
        print("Error: could not read image file.")
        return 2

    plate_region, bbox = detect_plate_region(image)
    if plate_region is None:
        print("No plate detected.")
        return 1

    extracted_plate = extract_text_from_plate(plate_region)
    if not extracted_plate:
        print("Plate detected, but OCR could not extract valid text.")
        return 1

    is_match = extracted_plate in allowset
    status = "SUCCESS" if is_match else "FAIL"

    print(f"Detected plate: {extracted_plate}")
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
    args = parser.parse_args()

    exit_code = process_image(Path(args.image), Path(args.allowlist), args.save_debug)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
