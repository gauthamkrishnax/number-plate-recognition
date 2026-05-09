# Number Plate Recognition

Python service to detect a number plate in an image and return the extracted text (OCR).

## Stacks and pipelines

**`app.py`** — full pipeline; choose with `--pipeline`:

- `auto` (default): ML PP-OCR first, then legacy contour + Tesseract fallback
- `ml`: PP-OCR only
- `legacy`: contour + Tesseract only

**`app2.py`** — faster staged RapidOCR on a downscaled image, then falls back to `app` (`auto`) when the fast path is unsure. Same debug overlay helpers as `app` via `app.save_plate_debug_image`.

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
python3 app2.py --image path/to/car.jpg --save-debug
```

### Batch test (`test/` → `output/`)

Runs every image in `test/`, writes `*_result.jpg` under `output/`, and writes timing stats to `output/batch_performance.json`.

```bash
python3 run_test_batch.py
```

Options:

- `--backend app2` (default) or `--backend app` — `app2` uses the fast path; `app` uses `app.py` only.
- `--pipeline auto|ml|legacy` — only applies when `--backend app` (same as `app.py --pipeline`).

## 3) gRPC server

Start the server (listens on all interfaces by default):

```bash
python3 grpc_server.py --host 0.0.0.0 --port 50051
```

Flags: `--host`, `--port`, `--workers` (thread pool size for concurrent RPCs).

### Service and messages

Package `plate`, service **`PlateRecognition`** (see `proto/plate_recognition.proto`):

| RPC | Request | Response |
|-----|---------|----------|
| `RecognizeFromPath` | `PathRequest` | `PlateResponse` |
| `RecognizeFromBytes` | `BytesRequest` | `PlateResponse` |

**`PathRequest`**

- `image_path` (string): filesystem path on the **server**; must be a readable file.
- `pipeline` (string): `auto`, `ml`, or `legacy`; empty or unset behaves like `auto`. **Used only when `backend` is `RECOGNITION_BACKEND_APP`.**
- `backend` (`RecognitionBackend`): which implementation runs (see below). Omit for default `APP`.

**`BytesRequest`**

- `image_data` (bytes): raw image file contents (e.g. JPEG/PNG).
- `pipeline` (string): same as path RPC; **only for `RECOGNITION_BACKEND_APP`.**
- `backend`: same as path RPC.

**`RecognitionBackend` enum**

- `RECOGNITION_BACKEND_APP` (0): `app.detect_and_read_plate` with the given `pipeline`.
- `RECOGNITION_BACKEND_APP2` (1): `app2.detect_and_read_plate_fast` (fast path, `auto` fallback inside `app2`).

**`PlateResponse`**

- `plate` (string): normalized plate text, or empty if none detected.
- `extraction_time_ms` (double): detection + OCR runtime inside the RPC, in milliseconds.
- `confidence_score` (double): backend quality/confidence score for the returned plate text (`0.0` if no plate).

### Python client example

```python
import grpc
import plate_recognition_pb2 as plate_pb2
import plate_recognition_pb2_grpc as plate_pb2_grpc

channel = grpc.insecure_channel("127.0.0.1:50051")
stub = plate_pb2_grpc.PlateRecognitionStub(channel)

# Default stack (app) + pipeline
r = stub.RecognizeFromPath(
    plate_pb2.PathRequest(image_path="/tmp/car.jpg", pipeline="auto")
)

# Fast stack (app2)
r2 = stub.RecognizeFromPath(
    plate_pb2.PathRequest(
        image_path="/tmp/car.jpg",
        pipeline="auto",  # ignored for APP2; may be omitted
        backend=plate_pb2.RECOGNITION_BACKEND_APP2,
    )
)
print(r.plate, r2.plate)
```

Errors are returned as gRPC status codes (for example `NOT_FOUND` for a missing path, `INVALID_ARGUMENT` for bad pipeline string or undecodable image).

## Notes

- Clear frontal images improve OCR.
- `rapidocr-onnxruntime` powers the ML stage.
