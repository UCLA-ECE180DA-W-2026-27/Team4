"""Route generic targets to Gemini and specific products to Qwen plus OCR.

The selected video frame is assumed to be stationary. Generic and specific
targets can be checked together in one command. Each result is saved as an
annotated image, with a combined summary image and a detailed JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

import gemini_video_check as gemini
import qwen_video_check as qwen_common
from qwen_mlx_video_check import (
    DEFAULT_MODEL as DEFAULT_QWEN_MODEL,
    LocalMlxQwenDetector,
    MacVisionTextVerifier,
)


DEFAULT_GENERIC_TARGETS = ["water", "cereal", "soup can", "ham"]


@dataclass(frozen=True)
class HybridResult:
    """One routed target-detection result."""

    target: str
    target_type: str
    backend: str
    inference_seconds: float
    box_2d: tuple[int, int, int, int] | None
    candidate_box_2d: tuple[int, int, int, int] | None
    usage: dict[str, int | None]
    raw_response: str | None
    ocr_text: str | None
    ocr_match: bool | None
    ocr_seconds: float | None
    error: str | None
    annotated_image: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Gemini for generic objects and local Qwen plus OCR for specific "
            "products on one stationary video frame."
        )
    )
    parser.add_argument("--video", type=Path, default=gemini.DEFAULT_VIDEO_PATH)
    parser.add_argument(
        "--generic-objects",
        nargs="+",
        default=None,
        help="Generic categories sent to Gemini, such as water or cereal.",
    )
    parser.add_argument(
        "--specific-objects",
        nargs="+",
        default=None,
        help="Named products sent to Qwen and verified with OCR.",
    )
    parser.add_argument("--time-seconds", type=float, default=26.0)
    parser.add_argument("--gemini-model", default=gemini.DEFAULT_MODEL)
    parser.add_argument("--gemini-upload-width", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-upload-width", type=int, default=960)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--ocr-fuzzy-threshold", type=float, default=0.74)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.generic_objects is None and args.specific_objects is None:
        args.generic_objects = list(DEFAULT_GENERIC_TARGETS)
        args.specific_objects = []
    else:
        args.generic_objects = args.generic_objects or []
        args.specific_objects = args.specific_objects or []
    args.generic_objects = [
        target.strip() for target in args.generic_objects if target.strip()
    ]
    args.specific_objects = [
        target.strip() for target in args.specific_objects if target.strip()
    ]
    if not args.generic_objects and not args.specific_objects:
        parser.error("provide at least one generic or specific object")
    if args.time_seconds < 0:
        parser.error("--time-seconds cannot be negative")
    if args.gemini_upload_width <= 0 or args.qwen_upload_width <= 0:
        parser.error("upload widths must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if not 0.0 <= args.ocr_fuzzy_threshold <= 1.0:
        parser.error("--ocr-fuzzy-threshold must be between 0 and 1")
    if not 0.0 <= args.ocr_min_confidence <= 1.0:
        parser.error("--ocr-min-confidence must be between 0 and 1")
    return args


def draw_result(
    frame: np.ndarray,
    timestamp_seconds: float,
    target: str,
    target_type: str,
    backend: str,
    inference_seconds: float,
    box_2d: tuple[int, int, int, int] | None,
    candidate_box_2d: tuple[int, int, int, int] | None,
    ocr_match: bool | None,
    error: str | None,
) -> np.ndarray:
    """Annotate one source-resolution result and its routed backend."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    header_height = 122
    cv2.rectangle(annotated, (0, 0), (width, header_height), (20, 20, 20), -1)
    if error:
        status = "DETECTION ERROR"
        color = (0, 0, 255)
    elif box_2d is not None:
        status = "TARGET DETECTED"
        color = (0, 220, 0)
    elif ocr_match is False:
        status = "QWEN CANDIDATE REJECTED BY OCR"
        color = (0, 165, 255)
    else:
        status = "NO TARGET DETECTED"
        color = (0, 180, 255)
    cv2.putText(
        annotated,
        f'{timestamp_seconds:.2f}s | {target_type.upper()} | target: "{target}"',
        (18, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"{status} | {backend}",
        (18, 71),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"End-to-end detection {inference_seconds:.2f}s",
        (18, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    displayed_box = box_2d
    box_label = f"{backend} target"
    if displayed_box is None and ocr_match is False:
        displayed_box = candidate_box_2d
        box_label = "OCR rejected Qwen candidate"
    if displayed_box is not None:
        left, top, right, bottom = qwen_common.pixel_box(
            displayed_box,
            width,
            height,
        )
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 4)
        cv2.putText(
            annotated,
            box_label,
            (left, max(header_height + 27, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    if error:
        cv2.putText(
            annotated,
            error[:150],
            (18, header_height + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def create_summary(images: list[np.ndarray]) -> np.ndarray:
    """Arrange result images into a two-column contact sheet."""
    columns = 2 if len(images) > 1 else 1
    rows = (len(images) + columns - 1) // columns
    cells = images + [np.zeros_like(images[0])] * (rows * columns - len(images))
    row_images = [
        cv2.hconcat(cells[index : index + columns])
        for index in range(0, len(cells), columns)
    ]
    summary = cv2.vconcat(row_images)
    if summary.shape[1] > 1920:
        scale = 1920 / summary.shape[1]
        summary = cv2.resize(
            summary,
            (1920, round(summary.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return summary


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")
    if args.generic_objects and not os.getenv("GEMINI_API_KEY"):
        raise SystemExit(
            "GEMINI_API_KEY is not set. Export it before checking generic objects."
        )
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else gemini.SCRIPT_DIRECTORY
        / "hybrid_object_test_output"
        / f"{gemini.safe_slug(video_path.stem)}_{args.time_seconds:g}s"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or source_width <= 0 or source_height <= 0:
        capture.release()
        raise SystemExit("Video reports invalid metadata")
    frame_index = min(frame_count - 1, round(args.time_seconds * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None or frame.size == 0:
        raise SystemExit(f"Could not read source frame {frame_index}")
    timestamp_seconds = frame_index / fps

    gemini_detector: gemini.GeminiTargetDetector | None = None
    if args.generic_objects:
        try:
            gemini_detector = gemini.GeminiTargetDetector(
                model=args.gemini_model,
                api_key=os.environ["GEMINI_API_KEY"],
                jpeg_quality=args.jpeg_quality,
                media_resolution="unspecified",
                upload_width=args.gemini_upload_width,
            )
        except Exception as error:
            raise SystemExit(f"Could not initialize Gemini: {error}") from error

    qwen_detector: LocalMlxQwenDetector | None = None
    ocr_verifier: MacVisionTextVerifier | None = None
    qwen_load_seconds: float | None = None
    if args.specific_objects:
        print("Loading Qwen once for all specific products...")
        try:
            qwen_detector = LocalMlxQwenDetector(
                model_id=args.qwen_model,
                upload_width=args.qwen_upload_width,
                max_new_tokens=args.max_new_tokens,
            )
            qwen_load_seconds = qwen_detector.load_seconds
            ocr_verifier = MacVisionTextVerifier(
                fuzzy_threshold=args.ocr_fuzzy_threshold,
                min_confidence=args.ocr_min_confidence,
            )
        except Exception as error:
            raise SystemExit(
                f"Could not initialize Qwen plus OCR: {error}"
            ) from error
        print(f"Loaded Qwen in {qwen_load_seconds:.2f}s.\n")

    print(f"Video: {video_path}")
    print(f"Stationary frame: {timestamp_seconds:.2f}s (frame {frame_index})")
    if args.generic_objects:
        print(
            "Generic -> Gemini: "
            f"{', '.join(args.generic_objects)} "
            f"({args.gemini_upload_width}px upload)"
        )
    if args.specific_objects:
        print(
            "Specific -> Qwen + OCR: "
            f"{', '.join(args.specific_objects)} "
            f"({args.qwen_upload_width}px Qwen input)"
        )
    print()

    result_rows: list[HybridResult] = []
    annotated_images: list[np.ndarray] = []
    batch_started = time.perf_counter()

    def save_result(
        target: str,
        target_type: str,
        backend: str,
        inference_seconds: float,
        box_2d: tuple[int, int, int, int] | None,
        candidate_box_2d: tuple[int, int, int, int] | None,
        usage: dict[str, int | None],
        raw_response: str | None,
        ocr_text: str | None,
        ocr_match: bool | None,
        ocr_seconds: float | None,
        error: str | None,
    ) -> None:
        annotated = draw_result(
            frame,
            timestamp_seconds,
            target,
            target_type,
            backend,
            inference_seconds,
            box_2d,
            candidate_box_2d,
            ocr_match,
            error,
        )
        sequence = len(result_rows) + 1
        image_path = output_directory / (
            f"{sequence:02d}_{target_type}_{gemini.safe_slug(target)}.jpg"
        )
        if not cv2.imwrite(str(image_path), annotated):
            raise RuntimeError(f"Could not write annotated image: {image_path}")
        annotated_images.append(annotated)
        result_rows.append(
            HybridResult(
                target=target,
                target_type=target_type,
                backend=backend,
                inference_seconds=inference_seconds,
                box_2d=box_2d,
                candidate_box_2d=candidate_box_2d,
                usage=usage,
                raw_response=raw_response,
                ocr_text=ocr_text,
                ocr_match=ocr_match,
                ocr_seconds=ocr_seconds,
                error=error,
                annotated_image=str(image_path),
            )
        )
        outcome = "error" if error else ("detected" if box_2d else "not detected")
        if ocr_match is False:
            outcome = "Qwen candidate rejected by OCR"
        print(f"{target_type:8} | {target}: {outcome}; {inference_seconds:.2f}s")

    assert (not args.generic_objects) or gemini_detector is not None
    for target in args.generic_objects:
        started = time.perf_counter()
        try:
            detections, usage_result = gemini_detector.detect(frame, target)
            error_message = None
        except KeyboardInterrupt:
            raise
        except Exception as error:
            detections = []
            usage_result = gemini.Usage(None, None, None)
            error_message = str(error)
        inference_seconds = time.perf_counter() - started
        save_result(
            target=target,
            target_type="generic",
            backend="Gemini",
            inference_seconds=inference_seconds,
            box_2d=detections[0].box_2d if detections else None,
            candidate_box_2d=None,
            usage=asdict(usage_result),
            raw_response=None,
            ocr_text=None,
            ocr_match=None,
            ocr_seconds=None,
            error=error_message,
        )

    assert (not args.specific_objects) or (
        qwen_detector is not None and ocr_verifier is not None
    )
    for target in args.specific_objects:
        started = time.perf_counter()
        ocr_seconds: float | None = None
        ocr_text: str | None = None
        ocr_match: bool | None = None
        raw_response = ""
        candidate_box: tuple[int, int, int, int] | None = None
        try:
            detections, usage_result, raw_response, _, _ = qwen_detector.detect(
                frame,
                target,
            )
            candidate_box = detections[0].box_2d if detections else None
            if detections:
                ocr_started = time.perf_counter()
                ocr_match, ocr_text = ocr_verifier.verify(
                    frame,
                    detections[0],
                    target,
                )
                ocr_seconds = time.perf_counter() - ocr_started
                if not ocr_match:
                    detections = []
            error_message = None
        except KeyboardInterrupt:
            raise
        except Exception as error:
            detections = []
            usage_result = qwen_common.Usage(None, None, None)
            error_message = str(error)
        inference_seconds = time.perf_counter() - started
        save_result(
            target=target,
            target_type="specific",
            backend="Qwen + OCR",
            inference_seconds=inference_seconds,
            box_2d=detections[0].box_2d if detections else None,
            candidate_box_2d=candidate_box,
            usage=asdict(usage_result),
            raw_response=raw_response,
            ocr_text=ocr_text,
            ocr_match=ocr_match,
            ocr_seconds=ocr_seconds,
            error=error_message,
        )

    batch_seconds = time.perf_counter() - batch_started
    summary_path = output_directory / "hybrid_summary.jpg"
    if not cv2.imwrite(str(summary_path), create_summary(annotated_images)):
        raise RuntimeError(f"Could not write summary image: {summary_path}")
    report = {
        "video": str(video_path),
        "stationary_frame": {
            "timestamp_seconds": timestamp_seconds,
            "frame_index": frame_index,
            "width": source_width,
            "height": source_height,
            "fps": fps,
        },
        "routing": {
            "generic": {
                "backend": "Gemini",
                "model": args.gemini_model,
                "upload_width": args.gemini_upload_width,
            },
            "specific": {
                "backend": "Qwen + OCR",
                "model": args.qwen_model,
                "qwen_upload_width": args.qwen_upload_width,
                "ocr_engine": "macOS Vision",
                "ocr_fuzzy_threshold": args.ocr_fuzzy_threshold,
                "ocr_min_confidence": args.ocr_min_confidence,
                "model_load_seconds": qwen_load_seconds,
            },
        },
        "batch_seconds_excluding_model_load": batch_seconds,
        "summary_image": str(summary_path),
        "results": [asdict(result) for result in result_rows],
    }
    results_path = output_directory / "results.json"
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nResults")
    print(f"Batch time excluding Qwen model load: {batch_seconds:.2f}s")
    if qwen_load_seconds is not None:
        print(f"Qwen model load time: {qwen_load_seconds:.2f}s")
    print(f"Summary image: {summary_path}")
    print(f"Detailed report: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
