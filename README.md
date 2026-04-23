# Number Plate Recognition App

Python app to:
- detect a number plate in an image
- extract the text using OCR
- compare it against an allowlist
- return success/fail

## 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need Tesseract OCR installed on your machine:

```bash
brew install tesseract
```

## 2) Add allowed plate values

Edit `allowed_plates.txt` and put one plate per line.

Example:

```text
MH12AB1234
KA01CD5678
DL8CAF5031
```

## 3) Run

```bash
python3 app.py --image path/to/car.jpg --allowlist allowed_plates.txt --save-debug
```

Output:
- `Detected plate: <VALUE>`
- `Match status: SUCCESS` if found in allowlist, otherwise `FAIL`

If `--save-debug` is used, a result image is created next to input image with a box around the detected plate.

## Notes

- For best OCR accuracy, use clear frontal images.
- This is a contour-based detector and works as a practical starter.
