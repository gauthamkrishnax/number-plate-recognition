# Number Plate Recognition

Python service to detect a number plate in an image and return the extracted text (OCR).

Pipelines:

- `auto` (default): ML PP-OCR first, then legacy contour + Tesseract fallback
- `ml`: PP-OCR only
- `legacy`: contour + Tesseract only

## 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Tesseract OCR on the host (used by the legacy path):

```bash
brew install tesseract
```

Regenerate gRPC Python stubs after editing `proto/plate_recognition.proto`:

```bash
python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/plate_recognition.proto
```

## 2) Command line

Prints only the plate string on success (exit code 0). Exit code 1 if no plate; 2 on I/O errors.

```bash
python3 app.py --image path/to/car.jpg
python3 app.py --image path/to/car.jpg --pipeline ml --save-debug
```

## 3) gRPC server

```bash
python3 grpc_server.py --host 0.0.0.0 --port 50051
```

RPCs (see `proto/plate_recognition.proto`):

- `RecognizeFromPath`: `image_path` on the **server** machine; optional `pipeline` (`auto` / `ml` / `legacy`).
- `RecognizeFromBytes`: raw file bytes (e.g. JPEG/PNG); optional `pipeline`.

Response message: `plate` — normalized plate text, or empty if none detected.

## Notes

- Clear frontal images improve OCR.
- `rapidocr-onnxruntime` powers the ML stage.
