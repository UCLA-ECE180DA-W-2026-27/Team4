"""Benchmark 4-bit Qwen3-VL product grounding with MLX on Apple Silicon.

This keeps the quantized model loaded across all sampled frames and reuses the
PyTorch benchmark's video sampling, box validation, annotation, and reporting
helpers so the latency and accuracy results are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
import unicodedata
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

import qwen_video_check as common


DEFAULT_MODEL = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
GENERIC_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"


class MacVisionTextVerifier:
    """Verify a Qwen candidate by reading its source-resolution crop."""

    GENERIC_CONTAINER_WORDS = {
        "bottle",
        "box",
        "can",
        "carton",
        "jar",
        "pack",
        "package",
        "product",
    }
    GENERIC_PRODUCT_WORDS = {
        "cereal",
        "ham",
        "soup",
        "tuna",
        "water",
    }

    def __init__(self, fuzzy_threshold: float, min_confidence: float) -> None:
        try:
            from Foundation import NSURL
            import Vision
        except ImportError as error:
            raise RuntimeError(
                "OCR verification requires the macOS Vision framework through "
                "PyObjC."
            ) from error
        self._url_class = NSURL
        self._vision = Vision
        self.fuzzy_threshold = fuzzy_threshold
        self.min_confidence = min_confidence

    @staticmethod
    def _tokens(value: str) -> list[str]:
        normalized = unicodedata.normalize("NFKD", value)
        normalized = normalized.replace("'", "").replace("’", "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").casefold()
        return re.findall(r"[a-z0-9]+", normalized)

    def _target_tokens(self, target: str) -> list[str]:
        tokens = [
            token
            for token in self._tokens(target)
            if token not in self.GENERIC_CONTAINER_WORDS
            and token not in self.GENERIC_PRODUCT_WORDS
        ]
        return tokens or self._tokens(target)

    def _word_matches(self, target: str, observed: str) -> bool:
        if target == observed:
            return True
        if (
            len(target) >= 5
            and len(observed) >= 4
            and (target in observed or observed in target)
        ):
            return True
        return SequenceMatcher(None, target, observed).ratio() >= self.fuzzy_threshold

    @staticmethod
    def _word_similarity(target: str, observed: str) -> float:
        return SequenceMatcher(None, target, observed).ratio()

    def recognize(
        self,
        frame: np.ndarray,
        detection: common.Detection,
    ) -> str:
        height, width = frame.shape[:2]
        left, top, right, bottom = common.pixel_box(
            detection.box_2d,
            width,
            height,
        )
        margin_x = max(8, round((right - left) * 0.08))
        margin_y = max(8, round((bottom - top) * 0.08))
        left = max(0, left - margin_x)
        right = min(width, right + margin_x)
        top = max(0, top - margin_y)
        bottom = min(height, bottom + margin_y)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            raise RuntimeError("Qwen returned an empty OCR crop")

        if crop.shape[1] < 700:
            scale = min(3.0, 700.0 / crop.shape[1])
            crop = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                temporary_path = Path(handle.name)
            if not cv2.imwrite(str(temporary_path), crop):
                raise RuntimeError("Could not create the temporary OCR crop")
            request = self._vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(
                None
            )
            request.setRecognitionLevel_(
                self._vision.VNRequestTextRecognitionLevelAccurate
            )
            request.setUsesLanguageCorrection_(True)
            request.setRecognitionLanguages_(["en-US"])
            handler = self._vision.VNImageRequestHandler.alloc().initWithURL_options_(
                self._url_class.fileURLWithPath_(str(temporary_path)),
                {},
            )
            succeeded, error = handler.performRequests_error_([request], None)
            if not succeeded:
                raise RuntimeError(f"macOS Vision OCR failed: {error}")
            lines: list[str] = []
            for observation in request.results() or []:
                candidates = observation.topCandidates_(1)
                if not candidates:
                    continue
                candidate = candidates[0]
                if float(candidate.confidence()) >= self.min_confidence:
                    lines.append(str(candidate.string()))
            return " | ".join(lines)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def verify(
        self,
        frame: np.ndarray,
        detection: common.Detection,
        target: str,
    ) -> tuple[bool, str]:
        text = self.recognize(frame, detection)
        observed_tokens = self._tokens(text)
        required_tokens = self._target_tokens(target)
        strong_match = bool(observed_tokens) and all(
            any(
                self._word_matches(required_token, observed_token)
                for observed_token in observed_tokens
            )
            for required_token in required_tokens
        )
        target_tokens = self._tokens(target)
        category_tokens = [
            token for token in target_tokens if token in self.GENERIC_PRODUCT_WORDS
        ]
        category_match = bool(category_tokens) and all(
            any(
                category_token == observed_token
                for observed_token in observed_tokens
            )
            for category_token in category_tokens
        )
        relaxed_threshold = max(0.60, self.fuzzy_threshold - 0.15)
        relaxed_distinctive_match = bool(observed_tokens) and all(
            any(
                len(observed_token) >= 4
                and self._word_similarity(required_token, observed_token)
                >= relaxed_threshold
                for observed_token in observed_tokens
            )
            for required_token in required_tokens
        )
        matched = strong_match or (category_match and relaxed_distinctive_match)
        return matched, text


class LocalMlxQwenDetector:
    """Load a 4-bit Qwen3-VL checkpoint once and run MLX Metal inference."""

    def __init__(
        self,
        model_id: str,
        upload_width: int | None,
        max_new_tokens: int,
    ) -> None:
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError as error:
            raise RuntimeError(
                "MLX Qwen requires mlx-vlm. Install it with "
                "`python -m pip install --upgrade mlx-vlm`."
            ) from error

        self.model_id = model_id
        self.device = "mlx-metal"
        self.dtype_name = "4-bit"
        self.upload_width = upload_width
        self.max_new_tokens = max_new_tokens
        self._generate = generate
        self._apply_chat_template = apply_chat_template

        load_started = time.perf_counter()
        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)
        self.load_seconds = time.perf_counter() - load_started

    def _prepare_image(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        image = frame
        height, width = image.shape[:2]
        if self.upload_width is not None and width > self.upload_width:
            scale = self.upload_width / width
            image = cv2.resize(
                image,
                (self.upload_width, round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        upload_height, upload_width = image.shape[:2]
        return image, upload_width, upload_height

    @staticmethod
    def _build_prompt(target: str, target_type: str) -> str:
        output_rules = (
            "Return only JSON in this exact format: "
            "{\"bbox_2d\":[x1,y1,x2,y2]}. Coordinates must be normalized "
            "integers from 0 to 1000. Stop after the first valid match. If no valid "
            "match is visible, return {\"bbox_2d\":[]}. Do not return a label, "
            "confidence, markdown, or explanation."
        )
        if target_type == "generic":
            return (
                f"Locate the first clearly visible object belonging to the general "
                f"category {target!r}. Any brand qualifies. Identify the category "
                "from the object's appearance, shape, packaging, and scene context; "
                "do not require matching brand text. Do not return a related product "
                "from a different category. "
                + output_rules
            )
        if target_type != "specific":
            raise ValueError("target_type must be 'generic' or 'specific'")
        return (
            f"Locate the first visible product matching {target!r}. Verify the brand "
            "and product using visible text, logo, and packaging; do not return a "
            "similar product from another brand. "
            + output_rules
        )

    def detect(
        self,
        frame: np.ndarray,
        target: str,
        target_type: str = "specific",
    ) -> tuple[list[common.Detection], common.Usage, str, int, int]:
        """Return the first target box, token counts, and raw model response."""
        image, image_width, image_height = self._prepare_image(frame)
        prompt = self._build_prompt(target, target_type)
        formatted_prompt = self._apply_chat_template(
            self.processor,
            self.config,
            prompt,
            num_images=1,
        )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
                temporary_path = Path(handle.name)
            if not cv2.imwrite(str(temporary_path), image):
                raise RuntimeError("Could not create the temporary MLX input image")
            result = self._generate(
                self.model,
                self.processor,
                formatted_prompt,
                [str(temporary_path)],
                max_tokens=self.max_new_tokens,
                temperature=0.0,
                verbose=False,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        response = result.text.strip()
        detections = common.parse_detection_payload(
            response,
            image_width,
            image_height,
        )
        usage = common.Usage(
            prompt_tokens=result.prompt_tokens,
            output_tokens=result.generation_tokens,
            total_tokens=result.total_tokens,
        )
        return detections, usage, response, image_width, image_height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 4-bit local Qwen3-VL grounding with MLX on sampled frames."
    )
    parser.add_argument("--video", type=Path, default=common.DEFAULT_VIDEO_PATH)
    parser.add_argument("--object", dest="target", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-every-seconds", type=float, default=2.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--upload-width", type=int, default=1280)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--ocr-verify",
        action="store_true",
        help="Reject Qwen boxes whose source-resolution crop lacks the target text.",
    )
    parser.add_argument("--ocr-fuzzy-threshold", type=float, default=0.74)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.25)
    parser.add_argument("--review-seconds", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    args.target = args.target.strip()
    if not args.target:
        parser.error("--object cannot be empty")
    if args.sample_every_seconds <= 0:
        parser.error("--sample-every-seconds must be positive")
    if args.start_seconds < 0:
        parser.error("--start-seconds cannot be negative")
    if args.end_seconds is not None and args.end_seconds <= args.start_seconds:
        parser.error("--end-seconds must be later than --start-seconds")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.upload_width is not None and args.upload_width <= 0:
        parser.error("--upload-width must be positive")
    if args.max_new_tokens <= 0 or args.review_seconds <= 0:
        parser.error("token and review limits must be positive")
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
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else common.DEFAULT_OUTPUT_ROOT
        / f"{common.safe_slug(video_path.stem)}_{common.safe_slug(args.target)}_mlx_4bit"
    )
    frames_directory = output_directory / "frames"
    output_video_path = (
        args.output_video.expanduser().resolve()
        if args.output_video is not None
        else output_directory / "qwen_mlx_sample_review.mp4"
    )
    frames_directory.mkdir(parents=True, exist_ok=True)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or width <= 0 or height <= 0 or fps <= 0:
        capture.release()
        raise SystemExit("Video reports invalid metadata")
    sample_indices = common.get_sample_indices(
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

    print(f"Video: {video_path}")
    print(f"Target: {args.target}")
    print(f"Local MLX model: {args.model}")
    print(f"Source: {width}x{height} at {fps:.2f} FPS")
    print(f"Sampled frames planned: {len(sample_indices)}")
    print("Loading the 4-bit MLX Qwen model once...")
    try:
        detector = LocalMlxQwenDetector(
            model_id=args.model,
            upload_width=args.upload_width,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as error:
        capture.release()
        raise SystemExit(f"Could not load local MLX Qwen: {error}") from error
    print(
        f"Loaded with MLX Metal in {detector.load_seconds:.1f}s. "
        "Model will be reused for all samples.\n"
    )
    ocr_verifier: MacVisionTextVerifier | None = None
    if args.ocr_verify:
        try:
            ocr_verifier = MacVisionTextVerifier(
                fuzzy_threshold=args.ocr_fuzzy_threshold,
                min_confidence=args.ocr_min_confidence,
            )
        except Exception as error:
            capture.release()
            raise SystemExit(f"Could not initialize OCR verification: {error}") from error
        print(
            "OCR verification enabled on source-resolution candidate crops "
            f"(fuzzy threshold {args.ocr_fuzzy_threshold:.2f}).\n"
        )

    results: list[common.FrameResult] = []
    annotated_frames: list[np.ndarray] = []
    upload_width = width
    upload_height = height
    benchmark_started = time.perf_counter()
    try:
        for sample_number, frame_index in enumerate(sample_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                error_message = f"OpenCV could not read source frame {frame_index}"
                if args.stop_on_error:
                    raise RuntimeError(error_message)
                result = common.FrameResult(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    inference_seconds=0.0,
                    detections=[],
                    usage=common.Usage(None, None, None),
                    raw_response="",
                    error=error_message,
                )
            else:
                started = time.perf_counter()
                candidate_detections: list[common.Detection] | None = None
                ocr_text: str | None = None
                ocr_match: bool | None = None
                ocr_seconds: float | None = None
                try:
                    detections, usage, raw, upload_width, upload_height = (
                        detector.detect(frame, args.target)
                    )
                    candidate_detections = list(detections)
                    if ocr_verifier is not None and candidate_detections:
                        ocr_started = time.perf_counter()
                        ocr_match, ocr_text = ocr_verifier.verify(
                            frame,
                            candidate_detections[0],
                            args.target,
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
                    error_message = str(error)
                inference_seconds = time.perf_counter() - started
                if error_message and args.stop_on_error:
                    raise RuntimeError(error_message)
                result = common.FrameResult(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    inference_seconds=inference_seconds,
                    detections=detections,
                    usage=usage,
                    raw_response=raw,
                    error=error_message,
                    candidate_detections=candidate_detections,
                    ocr_text=ocr_text,
                    ocr_match=ocr_match,
                    ocr_seconds=ocr_seconds,
                )
                annotated = common.draw_result(frame, result, args.target)
                annotated_frames.append(annotated)
                frame_path = frames_directory / (
                    f"frame_{frame_index:06d}_{frame_index / fps:07.2f}s.jpg"
                )
                if not cv2.imwrite(str(frame_path), annotated):
                    raise RuntimeError(f"Could not write annotated frame: {frame_path}")
            results.append(result)
            outcome = (
                f"ERROR: {result.error}"
                if result.error
                else (
                    "0 detection(s); OCR rejected candidate"
                    if result.ocr_match is False
                    else f"{len(result.detections)} detection(s)"
                )
            )
            ocr_suffix = (
                f"; OCR {result.ocr_seconds:.2f}s"
                if result.ocr_seconds is not None
                else ""
            )
            print(
                f"[{sample_number}/{len(sample_indices)}] "
                f"{result.timestamp_seconds:6.2f}s: {outcome}; "
                f"local MLX+verification {result.inference_seconds:.2f}s"
                f"{ocr_suffix}"
            )
    finally:
        capture.release()

    if not annotated_frames:
        raise SystemExit("No readable frames were available for output")
    common.create_review_video(
        annotated_frames,
        output_video_path,
        args.review_seconds,
    )
    elapsed_seconds = time.perf_counter() - benchmark_started
    successful = [result for result in results if result.error is None]
    report = {
        "video": str(video_path),
        "target": args.target,
        "provider": "local",
        "backend": "mlx",
        "model": args.model,
        "device": detector.device,
        "dtype": detector.dtype_name,
        "model_load_seconds": detector.load_seconds,
        "source": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        },
        "upload": {"width": upload_width, "height": upload_height},
        "sample_every_seconds": args.sample_every_seconds,
        "ocr_verification": {
            "enabled": args.ocr_verify,
            "engine": "macOS Vision" if args.ocr_verify else None,
            "fuzzy_threshold": args.ocr_fuzzy_threshold,
            "min_confidence": args.ocr_min_confidence,
            "rejected_candidates": sum(
                result.ocr_match is False for result in results
            ),
            "average_seconds": (
                sum(
                    result.ocr_seconds
                    for result in results
                    if result.ocr_seconds is not None
                )
                / sum(result.ocr_seconds is not None for result in results)
                if any(result.ocr_seconds is not None for result in results)
                else None
            ),
        },
        "sampled_frames_planned": len(sample_indices),
        "successful_inferences": len(successful),
        "frames_with_detections": sum(bool(result.detections) for result in results),
        "total_detections": sum(len(result.detections) for result in results),
        "total_benchmark_seconds_excluding_load": elapsed_seconds,
        "average_inference_seconds": (
            sum(result.inference_seconds for result in successful) / len(successful)
            if successful
            else None
        ),
        "frames": [asdict(result) for result in results],
    }
    results_path = output_directory / "results.json"
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nResults")
    print(f"Model load time: {detector.load_seconds:.1f}s")
    print(f"Successful local inferences: {len(successful)}/{len(results)}")
    print(f"Frames with detections: {report['frames_with_detections']}")
    if report["average_inference_seconds"] is not None:
        print(
            "Average local MLX+verification: "
            f"{report['average_inference_seconds']:.2f}s"
        )
    if report["ocr_verification"]["average_seconds"] is not None:
        print(
            "Average OCR verification: "
            f"{report['ocr_verification']['average_seconds']:.2f}s"
        )
        print(
            "OCR-rejected candidates: "
            f"{report['ocr_verification']['rejected_candidates']}"
        )
    print(f"Upload image size: {upload_width}x{upload_height}")
    print(f"Annotated frames: {frames_directory}")
    print(f"Review video: {output_video_path}")
    print(f"Detailed report: {results_path}")
    if not args.no_open:
        try:
            if common.open_video(output_video_path):
                print("Opened the review video.")
        except OSError as error:
            print(f"Could not open the review video automatically: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
