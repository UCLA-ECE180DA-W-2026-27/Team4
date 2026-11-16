"""Scan an entire video with Gemini generic and Qwen-plus-OCR specific routing.

The video is sampled at a configurable interval. Each target is retired after
its first confirmed detection, reducing local inference and Gemini API calls.
The script saves one final image per target and records every attempt's latency.
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
import hybrid_object_check as hybrid
import qwen_video_check as qwen_common
from qwen_mlx_video_check import (
    DEFAULT_MODEL as DEFAULT_QWEN_MODEL,
    LocalMlxQwenDetector,
    MacVisionTextVerifier,
)


DEFAULT_GENERIC_TARGETS = ["water", "cereal", "soup can", "ham"]


@dataclass(frozen=True)
class ScanAttempt:
    """One backend call for one target on one sampled frame."""

    target: str
    target_type: str
    backend: str
    frame_index: int
    timestamp_seconds: float
    inference_seconds: float
    detected: bool
    box_2d: tuple[int, int, int, int] | None
    candidate_box_2d: tuple[int, int, int, int] | None
    usage: dict[str, int | None]
    raw_response: str | None
    ocr_text: str | None
    ocr_match: bool | None
    ocr_seconds: float | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan sampled frames across a full video: Gemini handles generic "
            "objects and Qwen plus OCR handles specific products."
        )
    )
    parser.add_argument("--video", type=Path, default=gemini.DEFAULT_VIDEO_PATH)
    parser.add_argument("--generic-objects", nargs="+", default=None)
    parser.add_argument("--specific-objects", nargs="+", default=None)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, default=None)
    parser.add_argument("--sample-every-seconds", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=None)
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
    if args.start_seconds < 0:
        parser.error("--start-seconds cannot be negative")
    if args.end_seconds is not None and args.end_seconds <= args.start_seconds:
        parser.error("--end-seconds must be later than --start-seconds")
    if args.sample_every_seconds <= 0:
        parser.error("--sample-every-seconds must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
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


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")
    if args.generic_objects and not os.getenv("GEMINI_API_KEY"):
        raise SystemExit(
            "GEMINI_API_KEY is not set. Export it before scanning generic objects."
        )
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else gemini.SCRIPT_DIRECTORY
        / "hybrid_object_test_output"
        / f"{gemini.safe_slug(video_path.stem)}_full_scan"
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
    sample_indices = qwen_common.get_sample_indices(
        frame_count,
        fps,
        args.start_seconds,
        args.end_seconds,
        args.sample_every_seconds,
        args.max_samples,
    )
    if not sample_indices:
        capture.release()
        raise SystemExit("The selected video range contains no sampled frames")

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
            capture.release()
            raise SystemExit(f"Could not initialize Gemini: {error}") from error

    qwen_detector: LocalMlxQwenDetector | None = None
    ocr_verifier: MacVisionTextVerifier | None = None
    qwen_load_seconds: float | None = None
    if args.specific_objects:
        print("Loading Qwen once for all specific-product samples...")
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
            capture.release()
            raise SystemExit(
                f"Could not initialize Qwen plus OCR: {error}"
            ) from error
        print(f"Loaded Qwen in {qwen_load_seconds:.2f}s.\n")

    ordered_keys = [
        *(('generic', target) for target in args.generic_objects),
        *(('specific', target) for target in args.specific_objects),
    ]
    unresolved = set(ordered_keys)
    attempts: list[ScanAttempt] = []
    final_images: dict[tuple[str, str], np.ndarray] = {}
    final_image_paths: dict[tuple[str, str], str] = {}
    detection_attempts: dict[tuple[str, str], ScanAttempt] = {}
    processed_samples = 0
    scan_started = time.perf_counter()

    print(f"Video: {video_path}")
    print(
        f"Scanning {len(sample_indices)} sample(s) across the video at "
        f"{args.sample_every_seconds:g}-second intervals."
    )
    print("Each target stops after its first confirmed detection.\n")

    try:
        for sample_number, frame_index in enumerate(sample_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                print(f"[{sample_number}/{len(sample_indices)}] unreadable frame")
                continue
            processed_samples += 1
            timestamp_seconds = frame_index / fps
            print(
                f"[{sample_number}/{len(sample_indices)}] "
                f"{timestamp_seconds:.2f}s; unresolved targets: {len(unresolved)}"
            )

            assert (not args.generic_objects) or gemini_detector is not None
            for target in args.generic_objects:
                key = ("generic", target)
                if key not in unresolved:
                    continue
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
                box = detections[0].box_2d if detections else None
                attempt = ScanAttempt(
                    target=target,
                    target_type="generic",
                    backend="Gemini",
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    inference_seconds=inference_seconds,
                    detected=box is not None,
                    box_2d=box,
                    candidate_box_2d=None,
                    usage=asdict(usage_result),
                    raw_response=None,
                    ocr_text=None,
                    ocr_match=None,
                    ocr_seconds=None,
                    error=error_message,
                )
                attempts.append(attempt)
                annotated = hybrid.draw_result(
                    frame,
                    timestamp_seconds,
                    target,
                    "generic",
                    "Gemini",
                    inference_seconds,
                    box,
                    None,
                    None,
                    error_message,
                )
                final_images[key] = annotated
                if box is not None:
                    unresolved.remove(key)
                    detection_attempts[key] = attempt
                outcome = "detected" if box is not None else "not detected"
                if error_message:
                    outcome = f"error: {error_message}"
                print(f"  generic {target}: {outcome}; {inference_seconds:.2f}s")

            assert (not args.specific_objects) or (
                qwen_detector is not None and ocr_verifier is not None
            )
            for target in args.specific_objects:
                key = ("specific", target)
                if key not in unresolved:
                    continue
                started = time.perf_counter()
                ocr_seconds: float | None = None
                ocr_text: str | None = None
                ocr_match: bool | None = None
                raw_response = ""
                candidate_box: tuple[int, int, int, int] | None = None
                try:
                    detections, usage_result, raw_response, _, _ = (
                        qwen_detector.detect(frame, target)
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
                box = detections[0].box_2d if detections else None
                attempt = ScanAttempt(
                    target=target,
                    target_type="specific",
                    backend="Qwen + OCR",
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    inference_seconds=inference_seconds,
                    detected=box is not None,
                    box_2d=box,
                    candidate_box_2d=candidate_box,
                    usage=asdict(usage_result),
                    raw_response=raw_response,
                    ocr_text=ocr_text,
                    ocr_match=ocr_match,
                    ocr_seconds=ocr_seconds,
                    error=error_message,
                )
                attempts.append(attempt)
                annotated = hybrid.draw_result(
                    frame,
                    timestamp_seconds,
                    target,
                    "specific",
                    "Qwen + OCR",
                    inference_seconds,
                    box,
                    candidate_box,
                    ocr_match,
                    error_message,
                )
                final_images[key] = annotated
                if box is not None:
                    unresolved.remove(key)
                    detection_attempts[key] = attempt
                if box is not None:
                    outcome = "detected and OCR verified"
                elif ocr_match is False:
                    outcome = "candidate rejected by OCR"
                else:
                    outcome = "not detected"
                if error_message:
                    outcome = f"error: {error_message}"
                print(f"  specific {target}: {outcome}; {inference_seconds:.2f}s")

            if not unresolved:
                print("All targets were found; stopping the scan early.")
                break
    finally:
        capture.release()

    if not final_images:
        raise SystemExit("No readable sampled frames were available")
    result_images: list[np.ndarray] = []
    target_summaries: list[dict[str, object]] = []
    for sequence, key in enumerate(ordered_keys, start=1):
        target_type, target = key
        image = final_images[key]
        suffix = "detected" if key in detection_attempts else "not_found"
        image_path = output_directory / (
            f"{sequence:02d}_{target_type}_{gemini.safe_slug(target)}_{suffix}.jpg"
        )
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Could not write result image: {image_path}")
        final_image_paths[key] = str(image_path)
        result_images.append(image)
        target_attempts = [
            attempt
            for attempt in attempts
            if (attempt.target_type, attempt.target) == key
        ]
        detection = detection_attempts.get(key)
        target_summaries.append(
            {
                "target": target,
                "target_type": target_type,
                "backend": "Gemini" if target_type == "generic" else "Qwen + OCR",
                "detected": detection is not None,
                "first_detection_timestamp_seconds": (
                    detection.timestamp_seconds if detection is not None else None
                ),
                "first_detection_inference_seconds": (
                    detection.inference_seconds if detection is not None else None
                ),
                "attempt_count": len(target_attempts),
                "total_inference_seconds": sum(
                    attempt.inference_seconds for attempt in target_attempts
                ),
                "average_inference_seconds": (
                    sum(attempt.inference_seconds for attempt in target_attempts)
                    / len(target_attempts)
                    if target_attempts
                    else None
                ),
                "result_image": str(image_path),
            }
        )

    summary_path = output_directory / "hybrid_scan_summary.jpg"
    if not cv2.imwrite(str(summary_path), hybrid.create_summary(result_images)):
        raise RuntimeError(f"Could not write summary image: {summary_path}")
    scan_seconds = time.perf_counter() - scan_started
    report = {
        "video": str(video_path),
        "source": {
            "width": source_width,
            "height": source_height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps,
        },
        "sampling": {
            "start_seconds": args.start_seconds,
            "end_seconds": args.end_seconds,
            "sample_every_seconds": args.sample_every_seconds,
            "planned_samples": len(sample_indices),
            "processed_samples": processed_samples,
            "stopped_early": not unresolved,
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
                "ignored_generic_ocr_words": sorted(
                    MacVisionTextVerifier.GENERIC_PRODUCT_WORDS
                ),
                "model_load_seconds": qwen_load_seconds,
            },
        },
        "scan_seconds_excluding_qwen_load": scan_seconds,
        "summary_image": str(summary_path),
        "targets": target_summaries,
        "attempts": [asdict(attempt) for attempt in attempts],
    }
    results_path = output_directory / "results.json"
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nResults")
    for summary in target_summaries:
        status = "found" if summary["detected"] else "not found"
        timestamp = summary["first_detection_timestamp_seconds"]
        timestamp_text = f" at {timestamp:.2f}s" if timestamp is not None else ""
        print(
            f"{summary['target_type']:8} | {summary['target']}: "
            f"{status}{timestamp_text}; {summary['attempt_count']} attempt(s)"
        )
    print(f"Total scan time excluding Qwen load: {scan_seconds:.2f}s")
    if qwen_load_seconds is not None:
        print(f"Qwen model load time: {qwen_load_seconds:.2f}s")
    print(f"Summary image: {summary_path}")
    print(f"Detailed report: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
