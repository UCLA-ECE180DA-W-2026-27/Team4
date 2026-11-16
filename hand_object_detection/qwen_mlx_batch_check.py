"""Check several generic objects on one video frame with one Qwen model load.

OCR is intentionally not used. Each target gets its own annotated image, and a
combined summary image and JSON report make latency and detections easy to
compare.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import qwen_video_check as common
from qwen_mlx_video_check import GENERIC_MODEL, LocalMlxQwenDetector


DEFAULT_TARGETS = ["water", "cereal", "soup can", "ham"]
DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_COMPETITOR_CATEGORIES = (
    "water",
    "cereal",
    "soup can",
    "ham",
    "tuna",
    "chicken",
    "pasta",
    "snack food",
    "juice",
    "soda",
    "coffee",
    "tea",
)
CATEGORY_DESCRIPTIONS = {
    "water": "bottled drinking water",
    "cereal": "breakfast cereal in a box or bag",
    "soup can": "canned soup in a metal can",
    "ham": "packaged ham meat",
    "tuna": "packaged canned or pouched tuna fish",
    "chicken": "packaged chicken meat",
    "pasta": "packaged dry pasta or noodles",
    "snack food": "packaged snack food",
    "juice": "bottled or boxed fruit juice",
    "soda": "bottled or canned soda",
    "coffee": "packaged coffee",
    "tea": "packaged tea",
}


@dataclass(frozen=True)
class SiglipVerification:
    """One zero-shot classification decision for a Qwen crop."""

    matched: bool
    target_description: str
    target_score: float
    best_description: str
    best_score: float
    best_competitor_description: str
    best_competitor_score: float
    target_margin: float
    scores: dict[str, float]


class SiglipCropVerifier:
    """Verify Qwen crops with local SigLIP 2 zero-shot classification."""

    def __init__(self, model_id: str, minimum_margin: float) -> None:
        try:
            import torch
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError(
                "SigLIP verification requires torch, Pillow, and transformers."
            ) from error

        self.model_id = model_id
        self.minimum_margin = minimum_margin
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        load_started = time.perf_counter()
        self._classifier = pipeline(
            task="zero-shot-image-classification",
            model=model_id,
            device=self.device,
        )
        model_ready = time.perf_counter()
        from PIL import Image

        warmup_started = time.perf_counter()
        self._classifier(
            Image.new("RGB", (224, 224), color=(127, 127, 127)),
            candidate_labels=[
                CATEGORY_DESCRIPTIONS["cereal"],
                CATEGORY_DESCRIPTIONS["tuna"],
            ],
            hypothesis_template="This is a photo of {}.",
        )
        self.model_load_seconds = model_ready - load_started
        self.warmup_seconds = time.perf_counter() - warmup_started
        self.load_seconds = time.perf_counter() - load_started

    @staticmethod
    def _canonical_category(value: str) -> str:
        category = " ".join(value.casefold().strip().split())
        return {"cereral": "cereal", "soup": "soup can"}.get(category, category)

    @classmethod
    def _description(cls, category: str) -> str:
        canonical = cls._canonical_category(category)
        return CATEGORY_DESCRIPTIONS.get(
            canonical,
            f"a grocery store product belonging to the category {canonical}",
        )

    @staticmethod
    def _crop(
        frame: np.ndarray,
        detection: common.Detection,
        padding: float = 0.06,
    ) -> object:
        from PIL import Image

        height, width = frame.shape[:2]
        left, top, right, bottom = common.pixel_box(
            detection.box_2d,
            width,
            height,
        )
        pad_x = round((right - left) * padding)
        pad_y = round((bottom - top) * padding)
        left = max(0, left - pad_x)
        right = min(width, right + pad_x)
        top = max(0, top - pad_y)
        bottom = min(height, bottom + pad_y)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            raise RuntimeError("Qwen returned an empty SigLIP crop")
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    def verify(
        self,
        frame: np.ndarray,
        detection: common.Detection,
        target: str,
        competitor_categories: Sequence[str],
    ) -> SiglipVerification:
        target_category = self._canonical_category(target)
        categories = [target_category]
        for category in (*competitor_categories, *DEFAULT_COMPETITOR_CATEGORIES):
            canonical = self._canonical_category(category)
            if canonical not in categories:
                categories.append(canonical)
        descriptions = [self._description(category) for category in categories]
        target_description = descriptions[0]
        predictions = self._classifier(
            self._crop(frame, detection),
            candidate_labels=descriptions,
            hypothesis_template="This is a photo of {}.",
        )
        scores = {
            str(prediction["label"]): float(prediction["score"])
            for prediction in predictions
        }
        target_score = scores[target_description]
        competitor_scores = {
            description: score
            for description, score in scores.items()
            if description != target_description
        }
        best_description, best_score = max(scores.items(), key=lambda item: item[1])
        best_competitor_description, best_competitor_score = max(
            competitor_scores.items(),
            key=lambda item: item[1],
        )
        target_margin = target_score - best_competitor_score
        return SiglipVerification(
            matched=(
                best_description == target_description
                and target_margin >= self.minimum_margin
            ),
            target_description=target_description,
            target_score=target_score,
            best_description=best_description,
            best_score=best_score,
            best_competitor_description=best_competitor_description,
            best_competitor_score=best_competitor_score,
            target_margin=target_margin,
            scores=scores,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run several generic-object checks on one video frame while loading "
            "the local MLX Qwen model only once. OCR is not used."
        )
    )
    parser.add_argument("--video", type=Path, default=common.DEFAULT_VIDEO_PATH)
    parser.add_argument(
        "--objects",
        nargs="+",
        default=DEFAULT_TARGETS,
        help=(
            "Targets to check. Defaults to: water cereal 'soup can' ham. "
            "Quote targets containing spaces."
        ),
    )
    parser.add_argument(
        "--time-seconds",
        type=float,
        default=26.0,
        help="Video timestamp to analyze once for every target (default: 26).",
    )
    parser.add_argument(
        "--model",
        default=GENERIC_MODEL,
        help=(
            "MLX checkpoint (default: Qwen3-VL 4B 4-bit, selected for better "
            "generic-category grounding than the 2B model)."
        ),
    )
    parser.add_argument("--upload-width", type=int, default=960)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--siglip-verify",
        action="store_true",
        help="Verify every Qwen candidate crop with local SigLIP 2.",
    )
    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument(
        "--siglip-min-margin",
        type=float,
        default=0.0,
        help=(
            "Required target score advantage over the best competing category "
            "(default: 0, so the target must simply rank first)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    args.objects = [target.strip() for target in args.objects if target.strip()]
    if not args.objects:
        parser.error("--objects must contain at least one non-empty target")
    if args.time_seconds < 0:
        parser.error("--time-seconds cannot be negative")
    if args.upload_width <= 0:
        parser.error("--upload-width must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.siglip_min_margin < 0:
        parser.error("--siglip-min-margin cannot be negative")
    return args


def create_summary(images: list[np.ndarray]) -> np.ndarray:
    """Arrange annotated images in a compact grid."""
    if not images:
        raise ValueError("At least one image is required for a summary")
    columns = 2 if len(images) > 1 else 1
    rows = (len(images) + columns - 1) // columns
    height, width = images[0].shape[:2]
    blank = np.zeros_like(images[0])
    cells = images + [blank] * (rows * columns - len(images))
    row_images = [
        cv2.hconcat(cells[index : index + columns])
        for index in range(0, len(cells), columns)
    ]
    summary = cv2.vconcat(row_images)
    max_width = 1920
    if summary.shape[1] > max_width:
        scale = max_width / summary.shape[1]
        summary = cv2.resize(
            summary,
            (max_width, round(summary.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return summary


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else common.DEFAULT_OUTPUT_ROOT
        / (
            f"{common.safe_slug(video_path.stem)}_generic_batch_mlx_"
            f"{args.upload_width}"
        )
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

    print(f"Video: {video_path}")
    print(f"Timestamp: {timestamp_seconds:.2f}s (frame {frame_index})")
    print(f"Objects: {', '.join(args.objects)}")
    print(f"Local MLX model: {args.model}")
    print("Prompt mode: generic category (any brand qualifies)")
    print("OCR verification: disabled")
    print(f"SigLIP verification: {'enabled' if args.siglip_verify else 'disabled'}")
    print(f"Upload width: {args.upload_width}")
    print("Loading the 4-bit MLX Qwen model once...")
    try:
        detector = LocalMlxQwenDetector(
            model_id=args.model,
            upload_width=args.upload_width,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as error:
        raise SystemExit(f"Could not load local MLX Qwen: {error}") from error
    print(f"Loaded with MLX Metal in {detector.load_seconds:.2f}s.\n")

    siglip_verifier: SiglipCropVerifier | None = None
    if args.siglip_verify:
        print(f"Loading SigLIP verifier: {args.siglip_model}")
        try:
            siglip_verifier = SiglipCropVerifier(
                model_id=args.siglip_model,
                minimum_margin=args.siglip_min_margin,
            )
        except Exception as error:
            raise SystemExit(f"Could not load local SigLIP verifier: {error}") from error
        print(
            f"Loaded and warmed SigLIP on {siglip_verifier.device} in "
            f"{siglip_verifier.load_seconds:.2f}s "
            f"(warm-up {siglip_verifier.warmup_seconds:.2f}s).\n"
        )

    report_rows: list[dict[str, object]] = []
    annotated_images: list[np.ndarray] = []
    benchmark_started = time.perf_counter()
    upload_width = source_width
    upload_height = source_height
    for index, target in enumerate(args.objects, start=1):
        started = time.perf_counter()
        candidate_detections: list[common.Detection] | None = None
        siglip_result: SiglipVerification | None = None
        siglip_seconds: float | None = None
        try:
            qwen_started = time.perf_counter()
            detections, usage, raw, upload_width, upload_height = detector.detect(
                frame,
                target,
                target_type="generic",
            )
            qwen_seconds = time.perf_counter() - qwen_started
            candidate_detections = list(detections)
            if siglip_verifier is not None and candidate_detections:
                siglip_started = time.perf_counter()
                siglip_result = siglip_verifier.verify(
                    frame,
                    candidate_detections[0],
                    target,
                    args.objects,
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
        inference_seconds = time.perf_counter() - started
        result = common.FrameResult(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            inference_seconds=inference_seconds,
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
        annotated = common.draw_result(frame, result, target)
        annotated_images.append(annotated)
        image_path = output_directory / f"{index:02d}_{common.safe_slug(target)}.jpg"
        if not cv2.imwrite(str(image_path), annotated):
            raise RuntimeError(f"Could not write annotated image: {image_path}")
        report_rows.append(
            {
                "target": target,
                "detected": bool(detections),
                "inference_seconds": inference_seconds,
                "qwen_inference_seconds": qwen_seconds,
                "siglip_verification_seconds": siglip_seconds,
                "siglip": asdict(siglip_result) if siglip_result is not None else None,
                "qwen_candidate_detections": [
                    asdict(detection) for detection in (candidate_detections or [])
                ],
                "detections": [asdict(detection) for detection in detections],
                "usage": asdict(usage),
                "raw_response": raw,
                "error": error_message,
                "annotated_image": str(image_path),
            }
        )
        outcome = (
            f"ERROR: {error_message}"
            if error_message
            else (
                "detected; SigLIP matched"
                if siglip_result is not None and siglip_result.matched
                else (
                    "Qwen candidate rejected by SigLIP"
                    if siglip_result is not None
                    else ("detected" if detections else "not detected")
                )
            )
        )
        print(
            f"[{index}/{len(args.objects)}] {target}: {outcome}; "
            f"{inference_seconds:.2f}s total"
        )

    benchmark_seconds = time.perf_counter() - benchmark_started
    summary_path = output_directory / "batch_summary.jpg"
    if not cv2.imwrite(str(summary_path), create_summary(annotated_images)):
        raise RuntimeError(f"Could not write summary image: {summary_path}")
    successful_times = [
        float(row["inference_seconds"])
        for row in report_rows
        if row["error"] is None
    ]
    report = {
        "video": str(video_path),
        "timestamp_seconds": timestamp_seconds,
        "frame_index": frame_index,
        "targets": args.objects,
        "ocr_verification": False,
        "siglip_verification": {
            "enabled": args.siglip_verify,
            "model": args.siglip_model if args.siglip_verify else None,
            "device": siglip_verifier.device if siglip_verifier else None,
            "model_load_seconds": (
                siglip_verifier.load_seconds if siglip_verifier else None
            ),
            "warmup_seconds": (
                siglip_verifier.warmup_seconds if siglip_verifier else None
            ),
            "minimum_margin": args.siglip_min_margin,
            "competitor_categories": list(DEFAULT_COMPETITOR_CATEGORIES),
        },
        "model": args.model,
        "device": detector.device,
        "dtype": detector.dtype_name,
        "model_load_seconds": detector.load_seconds,
        "source": {
            "width": source_width,
            "height": source_height,
            "fps": fps,
            "frame_count": frame_count,
        },
        "upload": {"width": upload_width, "height": upload_height},
        "batch_seconds_excluding_load": benchmark_seconds,
        "average_inference_seconds": (
            sum(successful_times) / len(successful_times)
            if successful_times
            else None
        ),
        "summary_image": str(summary_path),
        "results": report_rows,
    }
    results_path = output_directory / "results.json"
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nResults")
    print(f"Model load time: {detector.load_seconds:.2f}s")
    if siglip_verifier is not None:
        print(
            f"SigLIP setup time: {siglip_verifier.load_seconds:.2f}s "
            f"(warm-up {siglip_verifier.warmup_seconds:.2f}s)"
        )
    print(f"{len(args.objects)}-object batch time: {benchmark_seconds:.2f}s")
    if report["average_inference_seconds"] is not None:
        print(f"Average inference: {report['average_inference_seconds']:.2f}s")
    print(f"Summary image: {summary_path}")
    print(f"Detailed report: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
