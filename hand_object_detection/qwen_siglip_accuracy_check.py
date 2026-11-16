"""Measure Qwen 4B plus SigLIP generic-object accuracy on labeled frames.

The default benchmark uses three clear frames from ``test1.mp4`` and evaluates
all four generic categories on each frame. Models are loaded once, every case
is annotated, and the report includes a confusion matrix and latency metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

import qwen_video_check as common
from qwen_mlx_batch_check import (
    DEFAULT_SIGLIP_MODEL,
    SiglipCropVerifier,
    create_summary,
)
from qwen_mlx_video_check import GENERIC_MODEL, LocalMlxQwenDetector


@dataclass(frozen=True)
class AccuracyCase:
    timestamp_seconds: float
    target: str
    expected_present: bool


DEFAULT_CASES = (
    AccuracyCase(0.0, "water", True),
    AccuracyCase(0.0, "cereal", False),
    AccuracyCase(0.0, "soup can", False),
    AccuracyCase(0.0, "ham", False),
    AccuracyCase(20.0, "water", False),
    AccuracyCase(20.0, "cereal", True),
    AccuracyCase(20.0, "soup can", False),
    AccuracyCase(20.0, "ham", False),
    AccuracyCase(26.0, "water", False),
    AccuracyCase(26.0, "cereal", False),
    AccuracyCase(26.0, "soup can", True),
    AccuracyCase(26.0, "ham", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Qwen 4B plus SigLIP accuracy on labeled stationary frames."
        )
    )
    parser.add_argument("--video", type=Path, default=common.DEFAULT_VIDEO_PATH)
    parser.add_argument("--model", default=GENERIC_MODEL)
    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--upload-width", type=int, default=960)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--siglip-min-margin", type=float, default=0.0)
    parser.add_argument(
        "--cases-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON list of {timestamp_seconds, target, expected_present}; "
            "otherwise use the 12 labeled test1.mp4 cases."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "generic_4b_960_siglip_accuracy",
    )
    args = parser.parse_args()
    if args.upload_width <= 0 or args.max_new_tokens <= 0:
        parser.error("width and token limits must be positive")
    if args.siglip_min_margin < 0:
        parser.error("--siglip-min-margin cannot be negative")
    return args


def load_cases(path: Path | None) -> list[AccuracyCase]:
    if path is None:
        return list(DEFAULT_CASES)
    resolved = path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read cases JSON: {error}") from error
    if not isinstance(payload, list) or not payload:
        raise SystemExit("Cases JSON must contain a non-empty list")
    cases: list[AccuracyCase] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise SystemExit(f"Case {index} must be a JSON object")
        try:
            timestamp = float(row["timestamp_seconds"])
            target = str(row["target"]).strip()
            expected = row["expected_present"]
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"Case {index} is invalid: {error}") from error
        if timestamp < 0 or not target or not isinstance(expected, bool):
            raise SystemExit(f"Case {index} has invalid values")
        cases.append(AccuracyCase(timestamp, target, expected))
    return cases


def load_frames(
    video_path: Path,
    cases: list[AccuracyCase],
) -> tuple[dict[float, tuple[int, float, np.ndarray]], dict[str, float | int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise SystemExit("Video reports invalid metadata")
    frames: dict[float, tuple[int, float, np.ndarray]] = {}
    try:
        for timestamp in sorted({case.timestamp_seconds for case in cases}):
            frame_index = min(frame_count - 1, round(timestamp * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise SystemExit(f"Could not read frame for {timestamp:.2f}s")
            frames[timestamp] = (frame_index, frame_index / fps, frame)
    finally:
        capture.release()
    metadata: dict[str, float | int] = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }
    return frames, metadata


def outcome_name(expected: bool, detected: bool) -> str:
    if expected and detected:
        return "TP"
    if expected:
        return "FN"
    if detected:
        return "FP"
    return "TN"


def add_ground_truth_banner(
    image: np.ndarray,
    expected: bool,
    outcome: str,
) -> np.ndarray:
    result = image.copy()
    color = (0, 210, 0) if outcome in {"TP", "TN"} else (0, 0, 255)
    label = (
        f"GROUND TRUTH: {'PRESENT' if expected else 'ABSENT'} | "
        f"OUTCOME: {outcome}"
    )
    cv2.rectangle(result, (0, 112), (result.shape[1], 154), (20, 20, 20), -1)
    cv2.putText(
        result,
        label,
        (18, 142),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    return result


def safe_divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")
    cases = load_cases(args.cases_json)
    frames, source = load_frames(video_path, cases)
    output_directory = args.output_dir.expanduser().resolve()
    frames_directory = output_directory / "cases"
    frames_directory.mkdir(parents=True, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"Accuracy cases: {len(cases)}")
    print(f"Qwen model: {args.model}")
    print(f"SigLIP model: {args.siglip_model}")
    print(f"Qwen upload width: {args.upload_width}")
    print("Loading and warming both models once...")
    try:
        qwen = LocalMlxQwenDetector(
            model_id=args.model,
            upload_width=args.upload_width,
            max_new_tokens=args.max_new_tokens,
        )
        siglip = SiglipCropVerifier(
            model_id=args.siglip_model,
            minimum_margin=args.siglip_min_margin,
        )
    except Exception as error:
        raise SystemExit(f"Could not initialize local models: {error}") from error
    print(
        f"Qwen ready in {qwen.load_seconds:.2f}s; SigLIP ready in "
        f"{siglip.load_seconds:.2f}s.\n"
    )

    targets = [case.target for case in cases]
    unique_targets = list(dict.fromkeys(targets))
    rows: list[dict[str, object]] = []
    annotated_images: list[np.ndarray] = []
    benchmark_started = time.perf_counter()
    for sequence, case in enumerate(cases, start=1):
        frame_index, actual_timestamp, frame = frames[case.timestamp_seconds]
        started = time.perf_counter()
        candidate_detections: list[common.Detection] = []
        siglip_result = None
        siglip_seconds: float | None = None
        try:
            qwen_started = time.perf_counter()
            detections, usage, raw, _, _ = qwen.detect(
                frame,
                case.target,
                target_type="generic",
            )
            qwen_seconds = time.perf_counter() - qwen_started
            candidate_detections = list(detections)
            if candidate_detections:
                siglip_started = time.perf_counter()
                siglip_result = siglip.verify(
                    frame,
                    candidate_detections[0],
                    case.target,
                    unique_targets,
                )
                siglip_seconds = time.perf_counter() - siglip_started
                if not siglip_result.matched:
                    detections = []
            error_message = None
        except KeyboardInterrupt:
            raise
        except Exception as error:
            detections = []
            usage = common.Usage(None, None, None)
            raw = ""
            qwen_seconds = time.perf_counter() - started
            error_message = str(error)
        total_seconds = time.perf_counter() - started
        detected = bool(detections)
        outcome = "ERROR" if error_message else outcome_name(
            case.expected_present,
            detected,
        )
        frame_result = common.FrameResult(
            frame_index=frame_index,
            timestamp_seconds=actual_timestamp,
            inference_seconds=total_seconds,
            detections=detections,
            usage=usage,
            raw_response=raw,
            error=error_message,
            candidate_detections=candidate_detections,
            verification_method="SigLIP" if siglip_result is not None else None,
            verification_match=(
                siglip_result.matched if siglip_result is not None else None
            ),
            verification_seconds=siglip_seconds,
        )
        annotated = common.draw_result(frame, frame_result, case.target)
        annotated = add_ground_truth_banner(
            annotated,
            case.expected_present,
            outcome,
        )
        annotated_images.append(annotated)
        image_path = frames_directory / (
            f"{sequence:02d}_{actual_timestamp:06.2f}s_"
            f"{common.safe_slug(case.target)}_{outcome.casefold()}.jpg"
        )
        if not cv2.imwrite(str(image_path), annotated):
            raise RuntimeError(f"Could not save annotated case: {image_path}")
        rows.append(
            {
                "sequence": sequence,
                "requested_timestamp_seconds": case.timestamp_seconds,
                "actual_timestamp_seconds": actual_timestamp,
                "frame_index": frame_index,
                "target": case.target,
                "expected_present": case.expected_present,
                "detected": detected,
                "outcome": outcome,
                "correct": outcome in {"TP", "TN"},
                "total_seconds": total_seconds,
                "qwen_seconds": qwen_seconds,
                "siglip_seconds": siglip_seconds,
                "qwen_candidates": [
                    asdict(detection) for detection in candidate_detections
                ],
                "accepted_detections": [
                    asdict(detection) for detection in detections
                ],
                "siglip": asdict(siglip_result) if siglip_result else None,
                "raw_qwen_response": raw,
                "error": error_message,
                "annotated_image": str(image_path),
            }
        )
        print(
            f"[{sequence:02d}/{len(cases)}] {actual_timestamp:5.2f}s | "
            f"{case.target:8} | expected "
            f"{'present' if case.expected_present else 'absent ':7} | "
            f"{outcome:5} | {total_seconds:.2f}s"
        )

    benchmark_seconds = time.perf_counter() - benchmark_started
    counts = {
        label: sum(row["outcome"] == label for row in rows)
        for label in ("TP", "TN", "FP", "FN", "ERROR")
    }
    evaluated = counts["TP"] + counts["TN"] + counts["FP"] + counts["FN"]
    positive_times = [
        float(row["total_seconds"])
        for row in rows
        if row["outcome"] == "TP"
    ]
    all_times = [
        float(row["total_seconds"])
        for row in rows
        if row["outcome"] != "ERROR"
    ]
    metrics = {
        "accuracy": safe_divide(counts["TP"] + counts["TN"], evaluated),
        "precision": safe_divide(counts["TP"], counts["TP"] + counts["FP"]),
        "recall": safe_divide(counts["TP"], counts["TP"] + counts["FN"]),
        "specificity": safe_divide(counts["TN"], counts["TN"] + counts["FP"]),
        "f1": safe_divide(2 * counts["TP"], 2 * counts["TP"] + counts["FP"] + counts["FN"]),
        "average_total_seconds": (
            sum(all_times) / len(all_times) if all_times else None
        ),
        "average_true_positive_seconds": (
            sum(positive_times) / len(positive_times) if positive_times else None
        ),
        "maximum_true_positive_seconds": max(positive_times, default=None),
    }
    summary_path = output_directory / "accuracy_summary.jpg"
    if not cv2.imwrite(str(summary_path), create_summary(annotated_images)):
        raise RuntimeError(f"Could not save summary image: {summary_path}")
    report = {
        "video": str(video_path),
        "source": source,
        "qwen": {
            "model": args.model,
            "upload_width": args.upload_width,
            "load_seconds": qwen.load_seconds,
        },
        "siglip": {
            "model": args.siglip_model,
            "load_seconds": siglip.load_seconds,
            "warmup_seconds": siglip.warmup_seconds,
            "minimum_margin": args.siglip_min_margin,
        },
        "case_count": len(cases),
        "confusion_matrix": counts,
        "metrics": metrics,
        "benchmark_seconds_excluding_model_setup": benchmark_seconds,
        "summary_image": str(summary_path),
        "cases": rows,
    }
    report_path = output_directory / "results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nAccuracy results")
    print(
        f"TP={counts['TP']} TN={counts['TN']} "
        f"FP={counts['FP']} FN={counts['FN']} errors={counts['ERROR']}"
    )
    for name in ("accuracy", "precision", "recall", "specificity", "f1"):
        value = metrics[name]
        print(f"{name.capitalize()}: {value:.1%}" if value is not None else f"{name.capitalize()}: n/a")
    if metrics["average_true_positive_seconds"] is not None:
        print(
            "Average correct-detection latency: "
            f"{metrics['average_true_positive_seconds']:.2f}s"
        )
        print(
            "Slowest correct-detection latency: "
            f"{metrics['maximum_true_positive_seconds']:.2f}s"
        )
    print(f"Summary image: {summary_path}")
    print(f"Detailed report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
