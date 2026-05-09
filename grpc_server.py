"""gRPC server: extract number plate text from an image path or raw image bytes."""

import argparse
from concurrent import futures
from pathlib import Path
import time
from typing import Any, Optional, Tuple

import cv2
import grpc
import numpy as np

import app
import app2
import plate_recognition_pb2
import plate_recognition_pb2_grpc

_VALID_PIPELINES = frozenset({"auto", "ml", "legacy"})


def _normalize_pipeline(raw: str) -> str:
    p = (raw or "").strip() or "auto"
    if p not in _VALID_PIPELINES:
        raise ValueError(f"pipeline must be one of {_VALID_PIPELINES}, got {raw!r}")
    return p


def _plate_from_image(
    image,
    *,
    pipeline: str,
    backend: int,
) -> Tuple[Optional[str], Any, float]:
    """Returns (plate_or_empty, bbox, confidence_score)."""
    if backend == plate_recognition_pb2.RECOGNITION_BACKEND_APP2:
        return app2.detect_and_read_plate_fast_with_score(image, fallback=True)
    return app.detect_and_read_plate_with_score(image, pipeline=pipeline)


class PlateRecognitionService(plate_recognition_pb2_grpc.PlateRecognitionServicer):
    def RecognizeFromPath(self, request, context):
        try:
            pipeline = _normalize_pipeline(request.pipeline)
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            return plate_recognition_pb2.PlateResponse()

        path = Path(request.image_path).expanduser()
        if not path.is_file():
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Image path does not exist or is not a file: {request.image_path}",
            )
            return plate_recognition_pb2.PlateResponse()

        image = cv2.imread(str(path))
        if image is None:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Could not decode image from path (unsupported or corrupt file).",
            )
            return plate_recognition_pb2.PlateResponse()

        t0 = time.perf_counter()
        plate, _, confidence_score = _plate_from_image(
            image,
            pipeline=pipeline,
            backend=request.backend,
        )
        extraction_time_ms = (time.perf_counter() - t0) * 1000.0
        return plate_recognition_pb2.PlateResponse(
            plate=plate or "",
            extraction_time_ms=extraction_time_ms,
            confidence_score=confidence_score if plate else 0.0,
        )

    def RecognizeFromBytes(self, request, context):
        try:
            pipeline = _normalize_pipeline(request.pipeline)
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            return plate_recognition_pb2.PlateResponse()

        if not request.image_data:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "image_data is empty.")
            return plate_recognition_pb2.PlateResponse()

        buf = np.frombuffer(request.image_data, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Could not decode image bytes (unsupported or corrupt data).",
            )
            return plate_recognition_pb2.PlateResponse()

        t0 = time.perf_counter()
        plate, _, confidence_score = _plate_from_image(
            image,
            pipeline=pipeline,
            backend=request.backend,
        )
        extraction_time_ms = (time.perf_counter() - t0) * 1000.0
        return plate_recognition_pb2.PlateResponse(
            plate=plate or "",
            extraction_time_ms=extraction_time_ms,
            confidence_score=confidence_score if plate else 0.0,
        )


def serve(host: str, port: int, max_workers: int) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    plate_recognition_pb2_grpc.add_PlateRecognitionServicer_to_server(
        PlateRecognitionService(), server
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"gRPC listening on {host}:{port}", flush=True)
    server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser(description="Number plate recognition gRPC server.")
    parser.add_argument("--host", default="[::]", help="Bind address (default [::]).")
    parser.add_argument("--port", type=int, default=50051, help="TCP port.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread pool size for concurrent RPC handling.",
    )
    args = parser.parse_args()
    serve(args.host, args.port, max(1, args.workers))


if __name__ == "__main__":
    main()
