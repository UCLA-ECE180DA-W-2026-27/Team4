"""Serve Qwen + SigLIP and optional OCR detections to Pi hand guidance.

The Raspberry Pi keeps ownership of the USB camera. It sends a resized JPEG
over the private USB-network connection, and this Mac server returns a
normalized bounding box for the requested object.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from hand_orientation import (
    DEFAULT_QWEN_MAX_NEW_TOKENS,
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_UPLOAD_WIDTH,
    DEFAULT_OCR_FUZZY_THRESHOLD,
    DEFAULT_OCR_MIN_CONFIDENCE,
    DEFAULT_SIGLIP_MODEL,
    QwenSiglipObjectDetector,
)


MAX_JPEG_BYTES = 10 * 1024 * 1024


class ObjectDetectionHTTPServer(ThreadingHTTPServer):
    """Serve detection and debug-preview requests on the private USB link."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        detector: QwenSiglipObjectDetector,
        target_name: str,
    ) -> None:
        super().__init__(address, ObjectDetectionRequestHandler)
        self.detector = detector
        self.target_name = target_name
        self.detection_lock = threading.Lock()
        self.preview_lock = threading.Lock()
        self.latest_preview: np.ndarray | None = None

    def set_preview(self, frame: np.ndarray) -> None:
        """Replace the displayed frame without blocking the Pi capture loop."""
        with self.preview_lock:
            self.latest_preview = frame

    def get_preview(self) -> np.ndarray | None:
        """Return the newest annotated Pi frame for the laptop window."""
        with self.preview_lock:
            return self.latest_preview


class ObjectDetectionRequestHandler(BaseHTTPRequestHandler):
    """Receive JPEG frames and return one normalized bounding box as JSON."""

    server: ObjectDetectionHTTPServer

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path != "/health":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "ready",
                "target": self.server.target_name,
                "ocr_verification": self.server.detector.ocr_enabled,
            },
        )

    def do_POST(self) -> None:
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path not in ("/detect", "/preview"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        if parsed_path.path == "/preview":
            frame = self._read_jpeg()
            if frame is None:
                return
            self.server.set_preview(frame)
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return

        requested_target = urllib.parse.parse_qs(parsed_path.query).get(
            "object",
            [""],
        )[0].strip()
        if requested_target.casefold() != self.server.target_name.casefold():
            self._send_json(
                HTTPStatus.CONFLICT,
                {
                    "error": "target_mismatch",
                    "server_target": self.server.target_name,
                    "requested_target": requested_target,
                },
            )
            return

        frame = self._read_jpeg()
        if frame is None:
            return

        started_at = time.perf_counter()
        try:
            with self.server.detection_lock:
                detection = self.server.detector.detect(frame)
        except Exception as error:
            self.log_error("Object detection failed: %s", error)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "detection_failed", "details": str(error)},
            )
            return
        latency_seconds = time.perf_counter() - started_at

        frame_height, frame_width = frame.shape[:2]
        if detection is None:
            payload = {
                "status": "not_detected",
                "target": self.server.target_name,
                "bbox_normalized": None,
                "latency_seconds": round(latency_seconds, 3),
                "source_size": [frame_width, frame_height],
            }
        else:
            left, top, box_width, box_height = detection.box
            payload = {
                "status": "detected",
                "target": self.server.target_name,
                "bbox_normalized": [
                    left / frame_width,
                    top / frame_height,
                    (left + box_width) / frame_width,
                    (top + box_height) / frame_height,
                ],
                "latency_seconds": round(latency_seconds, 3),
                "source_size": [frame_width, frame_height],
            }
        self._send_json(HTTPStatus.OK, payload)
        print(
            f"{payload['status']}: {self.server.target_name}; "
            f"{latency_seconds:.2f}s"
        )

    def _read_jpeg(self) -> np.ndarray | None:
        """Read and decode one bounded JPEG request body."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if not 0 < content_length <= MAX_JPEG_BYTES:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_jpeg_size"},
            )
            return None

        encoded_frame = self.rfile.read(content_length)
        frame = cv2.imdecode(
            np.frombuffer(encoded_frame, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None or frame.size == 0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_jpeg"},
            )
            return None
        return frame

    def log_message(self, format_string: str, *args) -> None:
        """Suppress high-frequency preview logs while retaining other requests."""
        if self.path.startswith("/preview"):
            return
        super().log_message(format_string, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen + SigLIP and optional Apple Vision OCR on the computer "
            "and serve boxes to the Pi."
        )
    )
    parser.add_argument(
        "--object",
        dest="object_name",
        required=True,
        help="Generic object requested by the user, such as 'cereal'",
    )
    parser.add_argument(
        "--host",
        required=True,
        help="Computer's private USB-network IP address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port used by the Pi (default: 8000)",
    )
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument(
        "--qwen-upload-width",
        type=int,
        default=DEFAULT_QWEN_UPLOAD_WIDTH,
    )
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=DEFAULT_QWEN_MAX_NEW_TOKENS,
    )
    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--siglip-min-margin", type=float, default=0.0)
    parser.add_argument(
        "--ocr-verify",
        action="store_true",
        help="Require Apple Vision OCR to match an exact branded product",
    )
    parser.add_argument(
        "--ocr-fuzzy-threshold",
        type=float,
        default=DEFAULT_OCR_FUZZY_THRESHOLD,
    )
    parser.add_argument(
        "--ocr-min-confidence",
        type=float,
        default=DEFAULT_OCR_MIN_CONFIDENCE,
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run the detector without the laptop debug window",
    )
    args = parser.parse_args()

    try:
        host_address = ipaddress.ip_address(args.host)
    except ValueError:
        parser.error("--host must be the computer's numeric USB-network IP")
    if not (host_address.is_private or host_address.is_link_local):
        parser.error("--host must be a private or link-local USB-network IP")
    if host_address.is_unspecified or host_address.is_loopback:
        parser.error("--host must identify the USB interface used by the Pi")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.qwen_upload_width <= 0:
        parser.error("--qwen-upload-width must be greater than zero")
    if args.qwen_max_new_tokens <= 0:
        parser.error("--qwen-max-new-tokens must be greater than zero")
    if args.siglip_min_margin < 0.0:
        parser.error("--siglip-min-margin cannot be negative")
    if not 0.0 <= args.ocr_fuzzy_threshold <= 1.0:
        parser.error("--ocr-fuzzy-threshold must be between 0 and 1")
    if not 0.0 <= args.ocr_min_confidence <= 1.0:
        parser.error("--ocr-min-confidence must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    target_name = args.object_name.strip()
    if not target_name:
        raise SystemExit("--object cannot be empty")

    detector = QwenSiglipObjectDetector(
        target_name=target_name,
        model_id=args.qwen_model,
        upload_width=args.qwen_upload_width,
        max_new_tokens=args.qwen_max_new_tokens,
        siglip_model_id=args.siglip_model,
        siglip_min_margin=args.siglip_min_margin,
        ocr_verify=args.ocr_verify,
        ocr_fuzzy_threshold=args.ocr_fuzzy_threshold,
        ocr_min_confidence=args.ocr_min_confidence,
    )
    server = ObjectDetectionHTTPServer(
        (args.host, args.port),
        detector,
        target_name,
    )
    print(f"Computer detector ready for target: {target_name}")
    print(
        "Apple Vision OCR verification: "
        f"{'enabled' if args.ocr_verify else 'disabled'}"
    )
    print(f"Health check: http://{args.host}:{args.port}/health")
    print(f"Pi endpoint: http://{args.host}:{args.port}/detect")
    print(f"Debug preview: http://{args.host}:{args.port}/preview")
    print("Press Q in the preview window or Ctrl-C in the terminal to stop.")

    server_thread = threading.Thread(
        target=server.serve_forever,
        name="object-detection-http",
        daemon=True,
    )
    server_thread.start()
    window_name = f"Pi Hand Guidance Debug - {target_name}"
    try:
        while server_thread.is_alive():
            if args.no_preview:
                time.sleep(0.1)
                continue
            preview_frame = server.get_preview()
            if preview_frame is None:
                # Do not spin in cv2.waitKey before the first Pi frame. On
                # macOS that can starve the background HTTP server's GIL time.
                time.sleep(0.05)
                continue
            cv2.imshow(window_name, preview_frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        print("\nStopping computer detector.")
    finally:
        server.shutdown()
        server_thread.join()
        server.server_close()
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
