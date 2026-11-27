"""Benchmark local Qwen3-VL product grounding on sampled video frames.

The model is loaded once and reused for every sample. Source frames remain at
their original resolution for saved annotations; ``--upload-width`` can resize
only the image passed to Qwen. No frames or prompts leave the computer.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Allow unsupported Metal operations to fall back to CPU instead of aborting.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = SCRIPT_DIRECTORY.parent / "test1.mp4"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIRECTORY / "qwen_video_test_output"
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


@dataclass(frozen=True)
class Detection:
    """One validated Qwen target box in normalized yxyx order."""

    box_2d: tuple[int, int, int, int]


@dataclass(frozen=True)
class Usage:
    """Local prompt and generated token counts."""

    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class FrameResult:
    """Local Qwen result for one sampled source frame."""

    frame_index: int
    timestamp_seconds: float
    inference_seconds: float
    detections: list[Detection]
    usage: Usage
    raw_response: str
    error: str | None = None
    candidate_detections: list[Detection] | None = None
    ocr_text: str | None = None
    ocr_match: bool | None = None
    ocr_seconds: float | None = None
    verification_method: str | None = None
    verification_match: bool | None = None
    verification_seconds: float | None = None


def build_grounding_prompt(target: str, target_type: str = "specific") -> str:
    """Build the shared grounding prompt used by every Qwen backend."""
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


def safe_slug(value: str) -> str:
    """Return a short filesystem-safe label."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug[:60] or "target"


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
    end_frame = (
        final_frame
        if end_seconds is None
        else min(final_frame, math.floor(end_seconds * fps))
    )
    step = max(1, round(sample_every_seconds * fps))
    indices = list(range(start_frame, end_frame + 1, step))
    return indices[:max_samples] if max_samples is not None else indices


def extract_json(value: str) -> object:
    """Extract a JSON value from a plain or fenced model response."""
    text = value.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        object_match = re.search(r"\{.*\}", text, re.DOTALL)
        array_match = re.search(r"\[[^\[\]]*\]", text, re.DOTALL)
        candidate = object_match.group(0) if object_match else (
            array_match.group(0) if array_match else text
        )
        return json.loads(candidate)


def parse_detection_payload(
    payload: str,
    image_width: int,
    image_height: int,
) -> list[Detection]:
    """Parse Qwen xyxy coordinates into normalized yxyx coordinates."""
    data = extract_json(payload)
    box: object
    if isinstance(data, list) and data and isinstance(data[0], dict):
        # Qwen sometimes wraps its one requested detection in a JSON array.
        # Accept that equivalent representation and continue using only the
        # first result, as required by the prompt.
        data = data[0]

    if isinstance(data, list):
        box = data
    elif isinstance(data, dict):
        box = (
            data.get("bbox_2d")
            or data.get("box_2d")
            or data.get("bbox")
            or data.get("box")
            or []
        )
    else:
        raise ValueError("Qwen response must contain a JSON box")
    if box == []:
        return []
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("Qwen box must be empty or contain four coordinates")
    try:
        x1, y1, x2, y2 = (float(coordinate) for coordinate in box)
    except (TypeError, ValueError) as error:
        raise ValueError("Qwen box coordinates must be numeric") from error

    # The prompt requests normalized coordinates. Some Qwen checkpoints still
    # return pixels, so convert values outside the normalized range defensively.
    if max(x1, x2) > 1000 or max(y1, y2) > 1000:
        x1 = x1 * 1000.0 / image_width
        x2 = x2 * 1000.0 / image_width
        y1 = y1 * 1000.0 / image_height
        y2 = y2 * 1000.0 / image_height

    xmin = max(0, min(1000, round(x1)))
    ymin = max(0, min(1000, round(y1)))
    xmax = max(0, min(1000, round(x2)))
    ymax = max(0, min(1000, round(y2)))
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Qwen returned an empty or reversed box")
    return [Detection(box_2d=(ymin, xmin, ymax, xmax))]


def pixel_box(
    box_2d: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    """Convert a normalized yxyx box into clamped pixel coordinates."""
    ymin, xmin, ymax, xmax = box_2d
    left = max(0, min(width - 1, round(xmin * width / 1000.0)))
    top = max(0, min(height - 1, round(ymin * height / 1000.0)))
    right = max(left + 1, min(width, round(xmax * width / 1000.0)))
    bottom = max(top + 1, min(height, round(ymax * height / 1000.0)))
    return left, top, right, bottom


class LocalQwenDetector:
    """Load Qwen3-VL once with PyTorch/Transformers on CUDA, MPS, or CPU."""

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype_name: str,
        upload_width: int | None,
        max_new_tokens: int,
        quantization: str = "none",
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForMultimodalLM as AutoQwenModel
            except ImportError:
                from transformers import AutoModelForImageTextToText as AutoQwenModel
        except ImportError as error:
            raise RuntimeError(
                "Local Qwen requires torch, torchvision, Pillow, and "
                "transformers>=4.57."
            ) from error

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Install a CUDA-enabled PyTorch build or use "
                "--qwen-device cpu."
            )
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "Metal is unavailable. Run from a normal Terminal session or use "
                "the CPU device."
            )
        if dtype_name == "auto":
            dtype_name = "float16" if device in ("cuda", "mps") else "float32"
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_by_name:
            raise ValueError(f"Unsupported Qwen dtype: {dtype_name}")
        dtype = dtype_by_name[dtype_name]
        if quantization == "auto":
            quantization = "4bit" if device == "cuda" else "none"
        if quantization not in ("none", "4bit"):
            raise ValueError("Qwen quantization must be 'auto', 'none', or '4bit'")
        if quantization == "4bit" and device != "cuda":
            raise RuntimeError(
                "4-bit Transformers Qwen currently requires a CUDA device in this "
                "application. Use --qwen-quantization none for CPU or MPS."
            )

        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype_name
        self.quantization = quantization
        self.upload_width = upload_width
        self.max_new_tokens = max_new_tokens
        self._torch = torch

        load_started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_id)
        model_kwargs: dict[str, object] = {"dtype": dtype}
        if quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as error:
                raise RuntimeError(
                    "Windows CUDA 4-bit Qwen requires bitsandbytes and accelerate. "
                    "Install them with `python -m pip install bitsandbytes "
                    "accelerate`."
                ) from error
            model_kwargs.update(
                {
                    "device_map": "auto",
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=dtype,
                    ),
                }
            )
        self.model = AutoQwenModel.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        if quantization == "none":
            self.model.to(device)
        self._synchronize()
        self.load_seconds = time.perf_counter() - load_started

    def _synchronize(self) -> None:
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        elif self.device == "mps":
            self._torch.mps.synchronize()

    def _prepare_image(self, frame: np.ndarray) -> tuple[object, int, int]:
        from PIL import Image

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if self.upload_width is not None and width > self.upload_width:
            scale = self.upload_width / width
            image = cv2.resize(
                image,
                (self.upload_width, round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        upload_height, upload_width = image.shape[:2]
        return Image.fromarray(image), upload_width, upload_height

    def detect(
        self,
        frame: np.ndarray,
        target: str,
        target_type: str = "specific",
    ) -> tuple[list[Detection], Usage, str, int, int]:
        """Return the first target box, token counts, and raw model response."""
        image, image_width, image_height = self._prepare_image(frame)
        prompt = build_grounding_prompt(target, target_type)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        prompt_tokens = int(inputs.input_ids.shape[-1])
        with self._torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        self._synchronize()
        trimmed = generated_ids[:, inputs.input_ids.shape[-1] :]
        output_tokens = int(trimmed.shape[-1])
        response = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        detections = parse_detection_payload(response, image_width, image_height)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
        )
        return detections, usage, response, image_width, image_height


def draw_result(
    frame: np.ndarray,
    result: FrameResult,
    target: str,
) -> np.ndarray:
    """Return a source frame annotated with the local Qwen result."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    header_height = 112
    cv2.rectangle(annotated, (0, 0), (width, header_height), (20, 20, 20), -1)

    if result.error:
        status = "LOCAL QWEN ERROR"
        status_color = (0, 0, 255)
    elif result.verification_match is False:
        status = (
            "QWEN CANDIDATE REJECTED BY "
            f"{(result.verification_method or 'VERIFIER').upper()}"
        )
        status_color = (0, 165, 255)
    elif result.ocr_match is False:
        status = "QWEN CANDIDATE REJECTED BY OCR"
        status_color = (0, 165, 255)
    elif result.detections:
        status = "QWEN FOUND TARGET"
        status_color = (0, 220, 0)
    else:
        status = "NO TARGET DETECTED"
        status_color = (0, 180, 255)
    cv2.putText(
        annotated,
        f'{result.timestamp_seconds:.2f}s | local Qwen3-VL | target: "{target}"',
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
    cv2.putText(
        annotated,
        (
            f"Local inference {result.inference_seconds:.2f}s"
            + (
                f" | OCR {'matched' if result.ocr_match else 'rejected'}"
                if result.ocr_match is not None
                else ""
            )
            + (
                f" | {result.verification_method or 'Verifier'} "
                f"{'matched' if result.verification_match else 'rejected'}"
                if result.verification_match is not None
                else ""
            )
        ),
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
            result.error[:140],
            (18, header_height + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated
    displayed_detections = (
        result.candidate_detections
        if (result.ocr_match is False or result.verification_match is False)
        and result.candidate_detections
        else result.detections
    )
    for detection in displayed_detections:
        left, top, right, bottom = pixel_box(detection.box_2d, width, height)
        rejected = result.ocr_match is False or result.verification_match is False
        box_color = (0, 165, 255) if rejected else (0, 220, 0)
        if result.verification_match is False:
            box_label = f"{result.verification_method or 'Verifier'} rejected Qwen box"
        elif result.ocr_match is False:
            box_label = "OCR rejected Qwen box"
        else:
            box_label = "Qwen target box"
        cv2.rectangle(annotated, (left, top), (right, bottom), box_color, 4)
        cv2.putText(
            annotated,
            box_label,
            (left, max(header_height + 27, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            box_color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def create_review_video(
    annotated_frames: list[np.ndarray],
    output_path: Path,
    seconds_per_frame: float,
) -> None:
    """Create a slideshow of independently analyzed samples."""
    height, width = annotated_frames[0].shape[:2]
    fps = 10.0
    repeats = max(1, round(fps * seconds_per_frame))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Qwen3-VL product grounding on sampled video frames."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--object", dest="target", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--quantization",
        choices=("auto", "4bit", "none"),
        default="auto",
        help="Use bitsandbytes 4-bit weights on CUDA (default: auto).",
    )
    parser.add_argument("--sample-every-seconds", type=float, default=2.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--upload-width", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
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
    return args


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
        / f"{safe_slug(video_path.stem)}_{safe_slug(args.target)}_qwen3_vl_2b"
    )
    frames_directory = output_directory / "frames"
    output_video_path = (
        args.output_video.expanduser().resolve()
        if args.output_video is not None
        else output_directory / "qwen_sample_review.mp4"
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

    print(f"Video: {video_path}")
    print(f"Target: {args.target}")
    print(f"Local model: {args.model}")
    print(f"Requested device: {args.device}; dtype: {args.dtype}")
    print(f"Source: {width}x{height} at {fps:.2f} FPS")
    print(f"Sampled frames planned: {len(sample_indices)}")
    print("Loading the local Qwen model once...")
    try:
        detector = LocalQwenDetector(
            model_id=args.model,
            device=args.device,
            dtype_name=args.dtype,
            upload_width=args.upload_width,
            max_new_tokens=args.max_new_tokens,
            quantization=args.quantization,
        )
    except Exception as error:
        capture.release()
        raise SystemExit(f"Could not load local Qwen: {error}") from error
    print(
        f"Loaded on {detector.device} in {detector.load_seconds:.1f}s. "
        "Model will be reused for all samples.\n"
    )

    results: list[FrameResult] = []
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
                result = FrameResult(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    inference_seconds=0.0,
                    detections=[],
                    usage=Usage(None, None, None),
                    raw_response="",
                    error=error_message,
                )
            else:
                started = time.perf_counter()
                try:
                    detections, usage, raw, upload_width, upload_height = detector.detect(
                        frame, args.target
                    )
                    error_message = None
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    detections = []
                    usage = Usage(None, None, None)
                    raw = ""
                    error_message = str(error)
                inference_seconds = time.perf_counter() - started
                if error_message and args.stop_on_error:
                    raise RuntimeError(error_message)
                result = FrameResult(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    inference_seconds=inference_seconds,
                    detections=detections,
                    usage=usage,
                    raw_response=raw,
                    error=error_message,
                )
                annotated = draw_result(frame, result, args.target)
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
                else f"{len(result.detections)} detection(s)"
            )
            print(
                f"[{sample_number}/{len(sample_indices)}] "
                f"{result.timestamp_seconds:6.2f}s: {outcome}; "
                f"local inference {result.inference_seconds:.2f}s"
            )
    finally:
        capture.release()

    if not annotated_frames:
        raise SystemExit("No readable frames were available for output")
    create_review_video(annotated_frames, output_video_path, args.review_seconds)
    elapsed_seconds = time.perf_counter() - benchmark_started
    successful = [result for result in results if result.error is None]
    report = {
        "video": str(video_path),
        "target": args.target,
        "provider": "local",
        "model": args.model,
        "device": detector.device,
        "dtype": detector.dtype_name,
        "quantization": detector.quantization,
        "model_load_seconds": detector.load_seconds,
        "source": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        },
        "upload": {"width": upload_width, "height": upload_height},
        "sample_every_seconds": args.sample_every_seconds,
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
        print(f"Average local inference: {report['average_inference_seconds']:.2f}s")
    print(f"Upload image size: {upload_width}x{upload_height}")
    print(f"Annotated frames: {frames_directory}")
    print(f"Review video: {output_video_path}")
    print(f"Detailed report: {results_path}")
    if not args.no_open:
        try:
            if open_video(output_video_path):
                print("Opened the review video.")
        except OSError as error:
            print(f"Could not open the review video automatically: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
