"""Check several objects on one video frame with one Qwen model load.

Optional PaddleOCR verification supports batches of branded products. Each
target gets its own annotated image, and a combined summary image and JSON
report make latency and detections easy to compare.
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
from qwen_mlx_video_check import (
    DEFAULT_MODEL as BRANDED_MODEL,
    GENERIC_MODEL,
    LocalMlxQwenDetector,
    PaddleTextVerifier,
    RapidTextVerifier,
)


DEFAULT_TARGETS = ["water", "cereal", "soup can", "ham"]
DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_OCR_FUZZY_THRESHOLD = 0.74
DEFAULT_OCR_MIN_CONFIDENCE = 0.25
DEFAULT_OCR_RESIZE_WIDTH = 480
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
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
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
            "Run several object checks on one video frame while loading the "
            "local MLX Qwen model and optional PaddleOCR verifier only once."
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
        default=None,
        help=(
            "MLX checkpoint. Defaults to Qwen3-VL 4B for generic categories "
            "and Qwen3-VL 2B when --ocr-verify is enabled."
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
    parser.add_argument(
        "--ocr-verify",
        action="store_true",
        help=(
            "Treat every target as a specific branded product and reject Qwen "
            "boxes whose source-resolution crop lacks matching text."
        ),
    )
    parser.add_argument(
        "--ocr-engine",
        choices=("paddle", "rapid", "rapid-v5"),
        default="paddle",
        help=(
            "Crop OCR backend used by --ocr-verify: PaddleOCR, RapidOCR's "
            "default PP-OCRv6 small model, or RapidOCR with English "
            "PP-OCRv5 mobile recognition (default: paddle)."
        ),
    )
    parser.add_argument(
        "--ocr-first",
        action="store_true",
        help=(
            "Run one full-frame PaddleOCR scan first, then run Qwen 2B only "
            "for targets whose distinctive text is present."
        ),
    )
    parser.add_argument(
        "--ocr-scan-width",
        type=int,
        default=960,
        help="Width of the full-frame OCR-first input (default: 960).",
    )
    parser.add_argument(
        "--specific-products",
        action="store_true",
        help=(
            "Use exact branded-product Qwen prompts without requiring OCR. "
            "This is implied by --ocr-verify."
        ),
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
        "--ocr-resize-width",
        type=int,
        default=DEFAULT_OCR_RESIZE_WIDTH,
        help=(
            "Minimum width used when enlarging a Qwen candidate before OCR "
            "(default: 480). Smaller values can reduce OCR latency."
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
    if args.ocr_scan_width <= 0:
        parser.error("--ocr-scan-width must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.siglip_min_margin < 0:
        parser.error("--siglip-min-margin cannot be negative")
    if not 0.0 <= args.ocr_fuzzy_threshold <= 1.0:
        parser.error("--ocr-fuzzy-threshold must be between 0 and 1")
    if not 0.0 <= args.ocr_min_confidence <= 1.0:
        parser.error("--ocr-min-confidence must be between 0 and 1")
    if args.ocr_resize_width <= 0:
        parser.error("--ocr-resize-width must be positive")
    if args.ocr_first and args.ocr_verify:
        parser.error("--ocr-first and --ocr-verify are mutually exclusive")
    if args.ocr_first and args.siglip_verify:
        parser.error("--ocr-first cannot be combined with --siglip-verify")
    if args.ocr_first and args.ocr_engine != "paddle":
        parser.error("--ocr-first currently supports only --ocr-engine paddle")
    if args.model is None:
        args.model = (
            BRANDED_MODEL if args.ocr_verify or args.ocr_first else GENERIC_MODEL
        )
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


def expanded_source_crop(
    text_box: tuple[int, int, int, int],
    scan_width: int,
    scan_height: int,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    """Expand an OCR text region into a product-sized source-frame crop."""
    left, top, right, bottom = text_box
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_width = min(
        scan_width,
        max((right - left) * 3.5, scan_width * 0.35),
    )
    crop_height = min(
        scan_height,
        max((bottom - top) * 5.0, scan_height * 0.55),
    )
    scan_left = max(0.0, min(scan_width - crop_width, center_x - crop_width / 2.0))
    scan_top = max(0.0, min(scan_height - crop_height, center_y - crop_height / 2.0))
    scan_right = scan_left + crop_width
    scan_bottom = scan_top + crop_height
    source_left = max(0, round(scan_left * source_width / scan_width))
    source_top = max(0, round(scan_top * source_height / scan_height))
    source_right = min(source_width, round(scan_right * source_width / scan_width))
    source_bottom = min(source_height, round(scan_bottom * source_height / scan_height))
    return source_left, source_top, source_right, source_bottom


def remap_crop_detection(
    detection: common.Detection,
    crop_box: tuple[int, int, int, int],
    source_width: int,
    source_height: int,
) -> common.Detection:
    """Map one normalized crop-relative Qwen box to the source frame."""
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    left, top, right, bottom = common.pixel_box(
        detection.box_2d,
        crop_width,
        crop_height,
    )
    return common.Detection(
        box_2d=(
            round((crop_top + top) * 1000.0 / source_height),
            round((crop_left + left) * 1000.0 / source_width),
            round((crop_top + bottom) * 1000.0 / source_height),
            round((crop_left + right) * 1000.0 / source_width),
        )
    )


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
    print(
        "Prompt mode: specific branded products"
        if args.specific_products or args.ocr_verify or args.ocr_first
        else "Prompt mode: generic category (any brand qualifies)"
    )
    ocr_engine_label = {
        "paddle": "PaddleOCR",
        "rapid": "RapidOCR ONNX",
        "rapid-v5": "RapidOCR English PP-OCRv5 ONNX",
    }[args.ocr_engine]
    print(
        f"{ocr_engine_label} verification: "
        f"{'enabled' if args.ocr_verify else 'disabled'}"
    )
    print(f"OCR-first gating: {'enabled' if args.ocr_first else 'disabled'}")
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

    ocr_verifier: PaddleTextVerifier | None = None
    ocr_load_seconds: float | None = None
    if args.ocr_verify or args.ocr_first:
        print(f"Loading {ocr_engine_label} verifier...")
        ocr_load_started = time.perf_counter()
        try:
            if args.ocr_engine == "paddle":
                ocr_verifier = PaddleTextVerifier(
                    fuzzy_threshold=args.ocr_fuzzy_threshold,
                    min_confidence=args.ocr_min_confidence,
                    resize_width=args.ocr_resize_width,
                )
            else:
                ocr_verifier = RapidTextVerifier(
                    fuzzy_threshold=args.ocr_fuzzy_threshold,
                    min_confidence=args.ocr_min_confidence,
                    resize_width=args.ocr_resize_width,
                    recognition_model=(
                        "v5-english-mobile"
                        if args.ocr_engine == "rapid-v5"
                        else "v6-small"
                    ),
                )
        except Exception as error:
            raise SystemExit(f"Could not load {ocr_engine_label} verifier: {error}") from error
        ocr_load_seconds = time.perf_counter() - ocr_load_started
        print(f"Loaded {ocr_engine_label} in {ocr_load_seconds:.2f}s.\n")

    report_rows: list[dict[str, object]] = []
    annotated_images: list[np.ndarray] = []
    benchmark_started = time.perf_counter()
    ocr_first_regions = []
    ocr_scan_seconds: float | None = None
    ocr_scan_width = source_width
    ocr_scan_height = source_height
    if args.ocr_first:
        assert ocr_verifier is not None
        ocr_frame = frame
        if source_width > args.ocr_scan_width:
            scale = args.ocr_scan_width / source_width
            ocr_frame = cv2.resize(
                frame,
                (args.ocr_scan_width, round(source_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ocr_scan_height, ocr_scan_width = ocr_frame.shape[:2]
        ocr_scan_started = time.perf_counter()
        ocr_first_regions = ocr_verifier.recognize_regions(ocr_frame)
        ocr_scan_seconds = time.perf_counter() - ocr_scan_started
        print(
            f"Full-frame OCR scan: {ocr_scan_width}x{ocr_scan_height}; "
            f"{len(ocr_first_regions)} text regions; {ocr_scan_seconds:.2f}s.\n"
        )
    upload_width = source_width
    upload_height = source_height
    for index, target in enumerate(args.objects, start=1):
        started = time.perf_counter()
        candidate_detections: list[common.Detection] | None = None
        siglip_result: SiglipVerification | None = None
        siglip_seconds: float | None = None
        ocr_text: str | None = None
        ocr_match: bool | None = None
        ocr_seconds: float | None = None
        ocr_first_crop_box: tuple[int, int, int, int] | None = None
        qwen_seconds = 0.0
        try:
            if args.ocr_first:
                assert ocr_verifier is not None
                assert ocr_scan_seconds is not None
                ocr_match, ocr_text, text_box = ocr_verifier.matching_region(
                    ocr_first_regions,
                    target,
                )
                ocr_seconds = ocr_scan_seconds
                detections = []
                candidate_detections = []
                usage = common.Usage(None, None, None)
                raw = ""
                if ocr_match:
                    if text_box is None:
                        ocr_first_crop_box = (0, 0, source_width, source_height)
                    else:
                        ocr_first_crop_box = expanded_source_crop(
                            text_box,
                            ocr_scan_width,
                            ocr_scan_height,
                            source_width,
                            source_height,
                        )
                    crop_left, crop_top, crop_right, crop_bottom = ocr_first_crop_box
                    qwen_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
                    qwen_started = time.perf_counter()
                    crop_detections, usage, raw, upload_width, upload_height = (
                        detector.detect(qwen_frame, target, target_type="specific")
                    )
                    qwen_seconds = time.perf_counter() - qwen_started
                    detections = [
                        remap_crop_detection(
                            detection,
                            ocr_first_crop_box,
                            source_width,
                            source_height,
                        )
                        for detection in crop_detections
                    ]
                    candidate_detections = list(detections)
            else:
                qwen_started = time.perf_counter()
                detections, usage, raw, upload_width, upload_height = detector.detect(
                    frame,
                    target,
                    target_type=(
                        "specific"
                        if args.specific_products or args.ocr_verify
                        else "generic"
                    ),
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
            if args.ocr_verify and ocr_verifier is not None and detections:
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
            usage = common.Usage(None, None, None)
            raw = ""
            qwen_seconds = time.perf_counter() - started
            error_message = str(error)
        inference_seconds = (
            (ocr_scan_seconds or 0.0) + qwen_seconds
            if args.ocr_first
            else time.perf_counter() - started
        )
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
            ocr_text=ocr_text,
            ocr_match=ocr_match,
            ocr_seconds=ocr_seconds,
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
                "ocr_verification_seconds": ocr_seconds,
                "ocr_match": ocr_match,
                "ocr_text": ocr_text,
                "ocr_first_crop_box": ocr_first_crop_box,
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
                "OCR text not found; Qwen skipped"
                if args.ocr_first and not ocr_match
                else (
                    "OCR matched; Qwen detected"
                    if args.ocr_first and detections
                    else "OCR matched; Qwen did not detect"
                )
            )
            if args.ocr_first
            else (
                f"Qwen candidate rejected by {ocr_engine_label}"
                if ocr_match is False
                else (
                    f"detected; {ocr_engine_label} matched"
                    if ocr_match is True
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
            )
        )
        ocr_suffix = (
            f"; OCR {ocr_seconds:.2f}s"
            + ("" if args.ocr_first else f"; text={ocr_text or '<none>'}")
            if ocr_seconds is not None
            else ""
        )
        print(
            f"[{index}/{len(args.objects)}] {target}: {outcome}; "
            f"{inference_seconds:.2f}s total{ocr_suffix}"
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
        "ocr_verification": {
            "enabled": args.ocr_verify or args.ocr_first,
            "mode": "ocr_first" if args.ocr_first else "qwen_then_ocr",
            "engine": (
                (
                    {
                        "rapid": "RapidOCR PP-OCRv6 Small ONNX",
                        "rapid-v5": (
                            "RapidOCR PP-OCRv6 detector + English "
                            "PP-OCRv5 Mobile recognizer ONNX"
                        ),
                        "paddle": "PaddleOCR PP-OCRv5 Mobile",
                    }[args.ocr_engine]
                )
                if args.ocr_verify or args.ocr_first
                else None
            ),
            "load_seconds": ocr_load_seconds,
            "full_frame_scan_seconds": ocr_scan_seconds,
            "full_frame_scan_width": ocr_scan_width if args.ocr_first else None,
            "full_frame_scan_height": ocr_scan_height if args.ocr_first else None,
            "fuzzy_threshold": args.ocr_fuzzy_threshold,
            "min_confidence": args.ocr_min_confidence,
            "resize_width": args.ocr_resize_width,
            "rejected_candidates": sum(
                row["ocr_match"] is False for row in report_rows
            ),
        },
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
    if ocr_load_seconds is not None:
        print(f"{ocr_engine_label} setup time: {ocr_load_seconds:.2f}s")
    if ocr_scan_seconds is not None:
        print(f"Full-frame OCR scan time: {ocr_scan_seconds:.2f}s")
    print(f"{len(args.objects)}-object batch time: {benchmark_seconds:.2f}s")
    if report["average_inference_seconds"] is not None:
        print(f"Average inference: {report['average_inference_seconds']:.2f}s")
    print(f"Summary image: {summary_path}")
    print(f"Detailed report: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
