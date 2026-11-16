"""Test direct periodic Gemini product detection on sampled video frames.

This benchmark is intentionally separate from the live hand-guidance program.
By default, one complete frame is sent directly to Gemini every two seconds
until the selected sample range ends. An optional YOLOE gate remains available
for comparison. The script writes annotated sample images and a review video
and records API latency/results in JSON. Analyzed frames are uploaded to the
Gemini API when this script is run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import chdir
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = SCRIPT_DIRECTORY.parent / "test1.mp4"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIRECTORY / "gemini_video_test_output"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_YOLO_MODEL_PATH = SCRIPT_DIRECTORY / "yoloe-26n-seg.pt"

DETECTION_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": (
        "The first valid target box as [ymin, xmin, ymax, xmax], normalized "
        "from 0 to 1000, or an empty array when the target is not visible."
    ),
    "items": {
        "type": "integer",
        "minimum": 0,
        "maximum": 1000,
    },
    "minItems": 0,
    "maxItems": 4,
}


@dataclass(frozen=True)
class Detection:
    """One validated Gemini target detection."""

    box_2d: tuple[int, int, int, int]


@dataclass(frozen=True)
class YoloCandidate:
    """One low-confidence YOLOE proposal in normalized coordinates."""

    box_2d: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class Usage:
    """Token counts reported by the Gemini SDK when available."""

    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class FrameResult:
    """YOLOE gate and optional Gemini result for one sampled source frame."""

    frame_index: int
    timestamp_seconds: float
    yolo_latency_seconds: float
    yolo_candidates: list[YoloCandidate]
    gemini_called: bool
    latency_seconds: float
    attempts: int
    detections: list[Detection]
    usage: Usage
    error: str | None = None


class YoloCandidateDetector:
    """Return every target proposal that passes a low YOLOE threshold."""

    def __init__(
        self,
        model_path: Path,
        target: str,
        input_size: int,
        confidence: float,
        iou: float,
        max_detections: int,
    ) -> None:
        try:
            import clip  # noqa: F401
            from ultralytics import YOLOE
        except ImportError as error:
            raise RuntimeError(
                "YOLOE gating requires ultralytics and its CLIP text encoder."
            ) from error

        self._input_size = input_size
        self._confidence = confidence
        self._iou = iou
        self._max_detections = max_detections
        self._model = YOLOE(str(model_path), verbose=False)

        # Ultralytics resolves mobileclip2_b.ts against the current directory.
        # Run setup beside this script so the existing local encoder is used.
        with chdir(SCRIPT_DIRECTORY):
            self._model.set_classes([target])

    def detect(self, frame: np.ndarray) -> list[YoloCandidate]:
        """Return target proposals as normalized [ymin, xmin, ymax, xmax]."""
        height, width = frame.shape[:2]
        results = self._model.predict(
            frame,
            imgsz=self._input_size,
            conf=self._confidence,
            iou=self._iou,
            device="cpu",
            max_det=self._max_detections,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        coordinates = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(np.int64)
        candidates: list[YoloCandidate] = []
        for coordinates_row, confidence, class_id in zip(
            coordinates, confidences, class_ids
        ):
            if int(class_id) != 0:
                continue
            left, top, right, bottom = coordinates_row
            xmin = max(0, min(1000, round(float(left) * 1000.0 / width)))
            ymin = max(0, min(1000, round(float(top) * 1000.0 / height)))
            xmax = max(0, min(1000, round(float(right) * 1000.0 / width)))
            ymax = max(0, min(1000, round(float(bottom) * 1000.0 / height)))
            if xmin >= xmax or ymin >= ymax:
                continue
            candidates.append(
                YoloCandidate(
                    box_2d=(ymin, xmin, ymax, xmax),
                    confidence=float(confidence),
                )
            )
        return candidates


class GeminiTargetDetector:
    """Send JPEG frames to Gemini and request structured target boxes."""

    def __init__(
        self,
        model: str,
        api_key: str,
        jpeg_quality: int,
        media_resolution: str,
        upload_width: int | None,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise RuntimeError(
                "Gemini detection requires google-genai. Install it with "
                "'python -m pip install google-genai'."
            ) from error

        self._model = model
        self._jpeg_quality = jpeg_quality
        self._upload_width = upload_width
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._media_resolution = (
            None
            if media_resolution == "unspecified"
            else {
                "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            }[media_resolution]
        )

    def detect(
        self,
        frame: np.ndarray,
        target: str,
        candidates: list[YoloCandidate] | None = None,
    ) -> tuple[list[Detection], Usage]:
        """Return the first valid target box and API token usage."""
        upload_frame = frame
        source_height, source_width = frame.shape[:2]
        if self._upload_width is not None and source_width > self._upload_width:
            scale = self._upload_width / source_width
            upload_frame = cv2.resize(
                frame,
                (self._upload_width, round(source_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            upload_frame,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        if not encoded_ok:
            raise RuntimeError("OpenCV could not encode the frame as JPEG")

        if candidates:
            candidate_lines = "\n".join(
                f"- candidate {index}: box_2d={list(candidate.box_2d)}, "
                f"YOLO confidence={candidate.confidence:.4f}"
                for index, candidate in enumerate(candidates, start=1)
            )
            context = (
                "A low-threshold local detector triggered this request. Its hints "
                "may be false positives and use [ymin, xmin, ymax, xmax] normalized "
                f"from 0 to 1000:\n{candidate_lines}\nTreat these only as hints "
                "and inspect the full frame because they may be imprecise. "
            )
        else:
            context = "Inspect the complete frame directly. "

        prompt = (
            f"Find the requested product {target!r}. {context}Verify that the target "
            "is truly visible. Return the first "
            "valid matching instance you identify and then stop. Return at most one "
            "detection. Do not search for or compare every matching instance, and do "
            "not select based on which one is clearest, largest, closest, or most "
            "centered. Any valid match is acceptable. Return only the requested target, "
            "not merely similar packages. If the request contains a brand or "
            "exact product name, verify it using visible label text, logo, and "
            "packaging when possible. If it is a general category, accept any product "
            "in that category. Omit uncertain candidates. The returned coordinates "
            "should tightly outline the entire visible product. Return only one JSON "
            "array containing [ymin, xmin, ymax, xmax], normalized from 0 to 1000. "
            "Do not return a label, confidence, key name, object, or explanation. "
            "Return [] when the target is not actually visible anywhere in the frame."
        )
        image_part = self._types.Part.from_bytes(
            data=encoded.tobytes(),
            mime_type="image/jpeg",
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=[image_part, prompt],
            config=self._types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=DETECTION_SCHEMA,
                # This is only a safety ceiling. A valid coordinate array normally
                # uses far fewer tokens, but 32 can truncate after model overhead.
                max_output_tokens=64,
                media_resolution=self._media_resolution,
                temperature=0.0,
            ),
        )

        response_text = (response.text or "").strip()
        if not response_text:
            raise RuntimeError("Gemini returned no JSON response")
        detections = parse_detection_payload(response_text)
        metadata = getattr(response, "usage_metadata", None)
        usage = Usage(
            prompt_tokens=_optional_int(metadata, "prompt_token_count"),
            output_tokens=_optional_int(metadata, "candidates_token_count"),
            total_tokens=_optional_int(metadata, "total_token_count"),
        )
        return detections, usage


def _optional_int(value: object, attribute: str) -> int | None:
    if value is None:
        return None
    raw_value = getattr(value, attribute, None)
    return int(raw_value) if raw_value is not None else None


def parse_detection_payload(payload: str | list[Any]) -> list[Detection]:
    """Parse and defensively validate Gemini's structured response."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(data, list):
        raise ValueError("Gemini response must be a JSON coordinate array")
    if not data:
        return []
    if len(data) != 4:
        raise ValueError("Gemini coordinate array must be empty or contain four values")
    try:
        ymin, xmin, ymax, xmax = (
            max(0, min(1000, round(float(coordinate)))) for coordinate in data
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Gemini coordinates must be numeric") from error
    if ymin >= ymax or xmin >= xmax:
        raise ValueError("Gemini returned an empty or reversed box")
    return [Detection(box_2d=(ymin, xmin, ymax, xmax))]


def pixel_box(
    box_2d: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    """Convert a normalized Gemini box into clamped pixel coordinates."""
    ymin, xmin, ymax, xmax = box_2d
    left = max(0, min(width - 1, round(xmin * width / 1000.0)))
    top = max(0, min(height - 1, round(ymin * height / 1000.0)))
    right = max(left + 1, min(width, round(xmax * width / 1000.0)))
    bottom = max(top + 1, min(height, round(ymax * height / 1000.0)))
    return left, top, right, bottom


def draw_result(
    frame: np.ndarray,
    result: FrameResult,
    target: str,
    yolo_gate: bool,
) -> np.ndarray:
    """Return a sample frame annotated with proposals and verified results."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    header_height = 112
    cv2.rectangle(annotated, (0, 0), (width, header_height), (20, 20, 20), -1)

    if result.error:
        status = "API ERROR"
        status_color = (0, 0, 255)
    elif result.detections:
        status = f"GEMINI CONFIRMED {len(result.detections)} TARGET(S)"
        status_color = (0, 220, 0)
    elif not result.gemini_called:
        status = "NO YOLO CANDIDATE - GEMINI SKIPPED"
        status_color = (0, 190, 255)
    else:
        status = "NO TARGET DETECTED"
        status_color = (0, 180, 255)

    mode_label = "YOLOE -> Gemini" if yolo_gate else "Direct Gemini sample"

    cv2.putText(
        annotated,
        f'{result.timestamp_seconds:.2f}s | {mode_label} | target: "{target}"',
        (18, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        status,
        (18, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        status_color,
        2,
        cv2.LINE_AA,
    )
    api_timing = (
        f"Gemini {result.latency_seconds:.2f}s"
        if result.gemini_called
        else "Gemini not called"
    )
    timing_text = (
        f"YOLOE {result.yolo_latency_seconds:.2f}s | {api_timing}"
        if yolo_gate
        else api_timing
    )
    cv2.putText(
        annotated,
        timing_text,
        (18, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    if result.error:
        cv2.putText(
            annotated,
            result.error[:120],
            (18, min(height - 20, header_height + 38)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    for candidate in result.yolo_candidates:
        left, top, right, bottom = pixel_box(candidate.box_2d, width, height)
        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 190, 255), 2)
        cv2.putText(
            annotated,
            f"YOLO candidate {candidate.confidence:.0%}",
            (left, max(header_height + 22, top - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 190, 255),
            2,
            cv2.LINE_AA,
        )

    for detection in result.detections:
        left, top, right, bottom = pixel_box(detection.box_2d, width, height)
        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 220, 0), 4)
        label = "Gemini target box"
        text_y = max(header_height + 27, top - 10)
        cv2.putText(
            annotated,
            label,
            (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def safe_slug(value: str) -> str:
    """Return a short filesystem-safe label."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug[:60] or "target"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send sampled video frames directly to Gemini for first-match product "
            "detection. Direct sampling defaults to every two seconds."
        )
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--object", dest="target", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--yolo-gate",
        action="store_true",
        help=(
            "Call Gemini only when low-threshold YOLOE produces a candidate; "
            "direct Gemini sampling is the default"
        ),
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=DEFAULT_YOLO_MODEL_PATH,
        help=f"YOLOE checkpoint (default: {DEFAULT_YOLO_MODEL_PATH})",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.05,
        help="Low YOLOE proposal threshold from 0 to 1 (default: 0.05)",
    )
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-input-size", type=int, default=960)
    parser.add_argument("--yolo-max-detections", type=int, default=20)
    parser.add_argument("--sample-every-seconds", type=float, default=2.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit sampled frames for a quick or low-cost test",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality used for Gemini uploads (default: 92)",
    )
    parser.add_argument(
        "--media-resolution",
        choices=("unspecified", "low", "medium", "high"),
        default="unspecified",
        help=(
            "Gemini image token budget (default: unspecified/model default). "
            "Use medium to test lower latency while retaining label detail."
        ),
    )
    parser.add_argument(
        "--upload-width",
        type=int,
        default=None,
        help=(
            "Downscale the Gemini upload to this width while preserving aspect "
            "ratio; saved source frames remain full resolution"
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help=(
            "Retry a failed API request this many times (default: 0). For periodic "
            "video sampling, the next fresh frame is preferable to retrying a stale one."
        ),
    )
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--review-seconds", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the generated review video when processing finishes",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop instead of recording an API error and continuing",
    )
    args = parser.parse_args()

    args.target = args.target.strip()
    if not args.target:
        parser.error("--object cannot be empty")
    args.model = args.model.strip()
    if not args.model:
        parser.error("--model cannot be empty")
    if not 0.0 < args.yolo_confidence <= 1.0:
        parser.error("--yolo-confidence must be greater than 0 and at most 1")
    if not 0.0 < args.yolo_iou <= 1.0:
        parser.error("--yolo-iou must be greater than 0 and at most 1")
    if args.yolo_input_size <= 0 or args.yolo_max_detections <= 0:
        parser.error("YOLOE input size and maximum detections must be positive")
    if args.sample_every_seconds <= 0.0:
        parser.error("--sample-every-seconds must be positive")
    if args.start_seconds < 0.0:
        parser.error("--start-seconds cannot be negative")
    if args.end_seconds is not None and args.end_seconds <= args.start_seconds:
        parser.error("--end-seconds must be later than --start-seconds")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be from 1 to 100")
    if args.upload_width is not None and args.upload_width <= 0:
        parser.error("--upload-width must be positive")
    if args.retries < 0 or args.retry_delay < 0.0:
        parser.error("--retries and --retry-delay cannot be negative")
    if args.review_seconds <= 0.0:
        parser.error("--review-seconds must be positive")
    return args


def get_sample_indices(
    frame_count: int,
    fps: float,
    start_seconds: float,
    end_seconds: float | None,
    sample_every_seconds: float,
    max_samples: int | None,
) -> list[int]:
    """Return unique source-frame indices selected by elapsed time."""
    final_frame = max(0, frame_count - 1)
    start_frame = min(final_frame, round(start_seconds * fps))
    if end_seconds is None:
        end_frame = final_frame
    else:
        end_frame = min(final_frame, math.floor(end_seconds * fps))
    step = max(1, round(sample_every_seconds * fps))
    indices = list(range(start_frame, end_frame + 1, step))
    if max_samples is not None:
        indices = indices[:max_samples]
    return indices


def create_review_video(
    annotated_frames: list[np.ndarray],
    output_path: Path,
    seconds_per_frame: float,
) -> None:
    """Create an honest sample-review slideshow instead of fake tracking."""
    if not annotated_frames:
        raise RuntimeError("No annotated frames are available for the review video")
    height, width = annotated_frames[0].shape[:2]
    review_fps = 10.0
    repeats = max(1, round(review_fps * seconds_per_frame))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        review_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create review video: {output_path}")
    try:
        for frame in annotated_frames:
            for _ in range(repeats):
                writer.write(frame)
    finally:
        writer.release()


def open_video(output_path: Path) -> bool:
    """Open a video with the operating system's default application."""
    if sys.platform == "darwin":
        command = ["open", str(output_path)]
    elif os.name == "nt":
        os.startfile(str(output_path))  # type: ignore[attr-defined]
        return True
    else:
        opener = shutil.which("xdg-open")
        if opener is None:
            return False
        command = [opener, str(output_path)]

    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return True


def main() -> int:
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Export it before running this benchmark."
        )

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")
    yolo_model_path = (
        args.yolo_model.expanduser().resolve() if args.yolo_gate else None
    )
    if yolo_model_path is not None and not yolo_model_path.is_file():
        raise SystemExit(f"YOLOE model not found: {yolo_model_path}")

    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
        / (
            f"{safe_slug(video_path.stem)}_{safe_slug(args.target)}_"
            f"{'yolo_gemini' if args.yolo_gate else 'direct_gemini'}"
        )
    )
    frames_directory = output_directory / "frames"
    output_video_path = (
        args.output_video.expanduser().resolve()
        if args.output_video is not None
        else output_directory / "gemini_sample_review.mp4"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    frames_directory.mkdir(parents=True, exist_ok=True)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise SystemExit("Video reports invalid dimensions or frame count")
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise SystemExit("Video reports an invalid frame rate")

    sample_indices = get_sample_indices(
        frame_count,
        fps,
        args.start_seconds,
        args.end_seconds,
        args.sample_every_seconds,
        args.max_samples,
    )
    if not sample_indices:
        capture.release()
        raise SystemExit("The selected time range contains no frames")

    yolo_detector: YoloCandidateDetector | None = None
    try:
        if yolo_model_path is not None:
            yolo_detector = YoloCandidateDetector(
                yolo_model_path,
                args.target,
                args.yolo_input_size,
                args.yolo_confidence,
                args.yolo_iou,
                args.yolo_max_detections,
            )
        gemini_detector = GeminiTargetDetector(
            args.model,
            api_key,
            args.jpeg_quality,
            args.media_resolution,
            args.upload_width,
        )
    except Exception as error:
        capture.release()
        raise SystemExit(f"Could not initialize detectors: {error}") from error

    print(f"Video: {video_path}")
    print(f"Target: {args.target}")
    if yolo_model_path is not None:
        print(
            f"Mode: YOLOE gate ({yolo_model_path.name} at "
            f"{args.yolo_confidence:.0%} confidence)"
        )
    else:
        print(
            f"Mode: direct Gemini sampling every "
            f"{args.sample_every_seconds:g} seconds"
        )
    print(f"Gemini model: {args.model}")
    print(f"Gemini media resolution: {args.media_resolution}")
    effective_upload_width = (
        min(width, args.upload_width) if args.upload_width is not None else width
    )
    effective_upload_height = round(height * effective_upload_width / width)
    print(
        f"Gemini upload size: {effective_upload_width}x{effective_upload_height} "
        f"at JPEG quality {args.jpeg_quality}"
    )
    print(f"Source: {width}x{height} at {fps:.2f} FPS")
    print(f"Sampled frames planned: {len(sample_indices)}")
    print(f"Maximum Gemini API calls: {len(sample_indices)}")
    if args.yolo_gate:
        print("Only frames with YOLOE candidates will be uploaded to Gemini.\n")
    else:
        print("Every readable sampled frame will be uploaded to Gemini.\n")

    started_at = time.perf_counter()
    results: list[FrameResult] = []
    annotated_frames: list[np.ndarray] = []
    try:
        for sample_number, frame_index in enumerate(sample_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                message = f"OpenCV could not read source frame {frame_index}"
                if args.stop_on_error:
                    raise RuntimeError(message)
                result = FrameResult(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    yolo_latency_seconds=0.0,
                    yolo_candidates=[],
                    gemini_called=False,
                    latency_seconds=0.0,
                    attempts=0,
                    detections=[],
                    usage=Usage(None, None, None),
                    error=message,
                )
                results.append(result)
                print(f"[{sample_number}/{len(sample_indices)}] {message}")
                continue

            timestamp_seconds = frame_index / fps
            candidates: list[YoloCandidate] = []
            yolo_latency_seconds = 0.0
            if yolo_detector is not None:
                yolo_started_at = time.perf_counter()
                candidates = yolo_detector.detect(frame)
                yolo_latency_seconds = time.perf_counter() - yolo_started_at
            detections: list[Detection] = []
            usage = Usage(None, None, None)
            error_message: str | None = None
            attempt_count = 0
            gemini_called = not args.yolo_gate or bool(candidates)
            latency_seconds = 0.0
            if gemini_called:
                api_started_at = time.perf_counter()
                for attempt in range(args.retries + 1):
                    attempt_count += 1
                    try:
                        detections, usage = gemini_detector.detect(
                            frame,
                            args.target,
                            candidates if args.yolo_gate else None,
                        )
                        error_message = None
                        break
                    except KeyboardInterrupt:
                        raise
                    except Exception as error:
                        error_message = str(error)
                        if attempt < args.retries:
                            time.sleep(args.retry_delay * (2**attempt))
                latency_seconds = time.perf_counter() - api_started_at

            if error_message and args.stop_on_error:
                raise RuntimeError(error_message)
            result = FrameResult(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                yolo_latency_seconds=yolo_latency_seconds,
                yolo_candidates=candidates,
                gemini_called=gemini_called,
                latency_seconds=latency_seconds,
                attempts=attempt_count,
                detections=detections,
                usage=usage,
                error=error_message,
            )
            results.append(result)
            annotated = draw_result(frame, result, args.target, args.yolo_gate)
            annotated_frames.append(annotated)
            frame_path = frames_directory / (
                f"frame_{frame_index:06d}_{timestamp_seconds:07.2f}s.jpg"
            )
            if not cv2.imwrite(str(frame_path), annotated):
                raise RuntimeError(f"Could not write annotated frame: {frame_path}")
            if error_message:
                outcome = f"ERROR: {error_message}"
            elif args.yolo_gate and gemini_called:
                outcome = (
                    f"{len(candidates)} YOLO candidate(s), "
                    f"{len(detections)} Gemini-confirmed"
                )
            elif args.yolo_gate:
                outcome = "no YOLO candidates; Gemini skipped"
            else:
                outcome = f"{len(detections)} Gemini detection(s)"
            timing = (
                f"YOLO {yolo_latency_seconds:.2f}s; Gemini {latency_seconds:.2f}s"
                if args.yolo_gate
                else f"Gemini {latency_seconds:.2f}s"
            )
            print(
                f"[{sample_number}/{len(sample_indices)}] "
                f"{timestamp_seconds:6.2f}s: {outcome}; {timing}; "
                f"{attempt_count} request attempt(s)"
            )
    finally:
        capture.release()

    if not annotated_frames:
        raise SystemExit("No readable frames were available for output")
    create_review_video(annotated_frames, output_video_path, args.review_seconds)

    elapsed_seconds = time.perf_counter() - started_at
    gemini_results = [result for result in results if result.gemini_called]
    successful_results = [
        result for result in gemini_results if result.error is None
    ]
    readable_results = [
        result
        for result in results
        if (
            result.gemini_called
            or result.yolo_latency_seconds > 0.0
            or result.yolo_candidates
        )
    ]
    frames_with_detections = sum(bool(result.detections) for result in results)
    total_detections = sum(len(result.detections) for result in results)
    total_yolo_candidates = sum(
        len(result.yolo_candidates) for result in results
    )
    total_api_attempts = sum(result.attempts for result in results)
    report = {
        "video": str(video_path),
        "target": args.target,
        "model": args.model,
        "source": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        },
        "sample_every_seconds": args.sample_every_seconds,
        "jpeg_quality": args.jpeg_quality,
        "media_resolution": args.media_resolution,
        "upload": {
            "width": effective_upload_width,
            "height": effective_upload_height,
        },
        "yolo_gate": {
            "enabled": args.yolo_gate,
            "model": str(yolo_model_path) if yolo_model_path is not None else None,
            "confidence": args.yolo_confidence,
            "iou": args.yolo_iou,
            "input_size": args.yolo_input_size,
            "max_detections": args.yolo_max_detections,
        },
        "sampled_frames_planned": len(sample_indices),
        "maximum_possible_api_calls": len(sample_indices),
        "gemini_calls_triggered": len(gemini_results),
        "frames_skipped_by_yolo_gate": sum(
            not result.gemini_called for result in readable_results
        ) if args.yolo_gate else 0,
        "total_yolo_candidates": total_yolo_candidates,
        "successful_api_calls": len(successful_results),
        "total_api_attempts": total_api_attempts,
        "frames_with_detections": frames_with_detections,
        "total_detections": total_detections,
        "total_processing_seconds": elapsed_seconds,
        "average_yolo_latency_seconds": (
            sum(result.yolo_latency_seconds for result in readable_results)
            / len(readable_results)
            if args.yolo_gate and readable_results
            else None
        ),
        "average_api_latency_seconds": (
            sum(result.latency_seconds for result in successful_results)
            / len(successful_results)
            if successful_results
            else None
        ),
        "frames": [asdict(result) for result in results],
    }
    results_path = output_directory / "results.json"
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nResults")
    print(f"Sampled frames processed: {len(readable_results)}/{len(sample_indices)}")
    if args.yolo_gate:
        print(f"Total YOLOE candidates: {total_yolo_candidates}")
    print(f"Gemini calls triggered: {len(gemini_results)}")
    print(f"Successful Gemini calls: {len(successful_results)}")
    print(f"Actual Gemini request attempts: {total_api_attempts}")
    print(f"Frames with confirmed detections: {frames_with_detections}")
    print(f"Total confirmed detections: {total_detections}")
    print(f"Total processing time: {elapsed_seconds:.1f}s")
    print(f"Annotated frames: {frames_directory}")
    print(f"Review video: {output_video_path}")
    print(f"Detailed report: {results_path}")
    if not args.no_open:
        try:
            if open_video(output_video_path):
                print("Opened the review video.")
            else:
                print("Could not find a system video opener; open the path above.")
        except OSError as error:
            print(f"Could not open the review video automatically: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
