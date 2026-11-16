"""Run an isolated YOLOE object-detection check on a prerecorded video.

This script intentionally does not import MediaPipe or open a camera. It reads
frames from a video, searches for one open-vocabulary target, writes an
annotated MP4, and reports how often the target was detected.
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = SCRIPT_DIRECTORY.parent / "IMG_5095.MOV"
DEFAULT_MODEL_PATH = SCRIPT_DIRECTORY / "yoloe-26n-seg.pt"
DEFAULT_OUTPUT_PATH = (
    SCRIPT_DIRECTORY / "video_test_output" / "IMG_5095_detected.mp4"
)


@dataclass(frozen=True)
class Detection:
    """The strongest target detection in one video frame."""

    box: tuple[int, int, int, int]
    confidence: float


class ObjectDetector:
    """Minimal YOLOE detector with no hand-tracking dependencies."""

    def __init__(
        self,
        model_path: Path,
        target: str,
        distractors: Sequence[str],
        reference_image: str | np.ndarray | None,
        reference_boxes: Sequence[Sequence[float]],
        input_size: int,
        confidence: float,
        iou: float,
    ) -> None:
        try:
            from ultralytics import YOLOE
        except ImportError as error:
            raise RuntimeError(
                "Install ultralytics in the active environment."
            ) from error

        self._input_size = input_size
        self._confidence = confidence
        self._iou = iou
        self._model = YOLOE(str(model_path), verbose=False)
        self._reference_image = reference_image
        self._visual_prompts: dict[str, np.ndarray] | None = None

        if reference_image is not None:
            boxes = np.asarray(reference_boxes, dtype=np.float32)
            self._visual_prompts = {
                "bboxes": boxes,
                "cls": np.zeros(len(boxes), dtype=np.int64),
            }
        else:
            try:
                import clip  # noqa: F401
            except ImportError as error:
                raise RuntimeError(
                    "Text prompting requires the Ultralytics CLIP text encoder."
                ) from error
            # Ultralytics resolves mobileclip2_b.ts against the current directory.
            # Run setup beside this script so the existing local encoder is used.
            with chdir(SCRIPT_DIRECTORY):
                self._model.set_classes([target, *distractors])

    def detect(self, frame: np.ndarray) -> Detection | None:
        """Return the highest-confidence target detection in a frame."""
        frame_height, frame_width = frame.shape[:2]
        predict_arguments = {
            "imgsz": self._input_size,
            "conf": self._confidence,
            "iou": self._iou,
            "device": "cpu",
            "max_det": 20,
            "verbose": False,
        }
        if self._visual_prompts is not None:
            results = self._model.predict(
                frame,
                refer_image=self._reference_image,
                visual_prompts=self._visual_prompts,
                **predict_arguments,
            )
            # The reference sets a persistent embedding for later frames.
            self._visual_prompts = None
            self._reference_image = None
        else:
            results = self._model.predict(frame, **predict_arguments)
        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        confidences = boxes.conf.detach().cpu().numpy()
        coordinates = boxes.xyxy.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(np.int64)
        target_indices = np.flatnonzero(class_ids == 0)
        if target_indices.size == 0:
            return None
        best_index = int(
            target_indices[int(np.argmax(confidences[target_indices]))]
        )
        left_float, top_float, right_float, bottom_float = coordinates[best_index]

        left = max(0, min(int(math.floor(left_float)), frame_width - 1))
        top = max(0, min(int(math.floor(top_float)), frame_height - 1))
        right = max(left + 1, min(int(math.ceil(right_float)), frame_width))
        bottom = max(top + 1, min(int(math.ceil(bottom_float)), frame_height))
        return Detection(
            box=(left, top, right - left, bottom - top),
            confidence=float(confidences[best_index]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test YOLOE object detection on a prerecorded video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help=f"Input video (default: {DEFAULT_VIDEO_PATH})",
    )
    parser.add_argument(
        "--object",
        dest="target",
        required=True,
        help='Target description, for example --object "pasta box"',
    )
    parser.add_argument(
        "--distractor",
        action="append",
        default=[],
        help=(
            "Competing class that should not count as the target; repeat this "
            "option for multiple classes"
        ),
    )
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--reference-image",
        type=Path,
        default=None,
        help="Image containing one or more visual examples of the target",
    )
    reference_group.add_argument(
        "--reference-frame",
        type=int,
        default=None,
        help="Zero-based frame from --video containing visual target examples",
    )
    parser.add_argument(
        "--reference-box",
        type=float,
        nargs=4,
        action="append",
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=[],
        help=(
            "Target box in reference-image pixels; repeat for multiple "
            "examples of the same target"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"YOLOE checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Annotated MP4 path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help=(
            "Minimum confidence from 0 to 1 (default: 0.25 for text or "
            "0.05 for a visual prompt)"
        ),
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Overlap-suppression threshold from 0 to 1 (default: 0.45)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=960,
        help="Square inference size in pixels (default: 960)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=5,
        help="Run detection every N frames (default: 5; use 1 for every frame)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames; useful for a quick check",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Begin at this zero-based video frame (default: 0)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the annotated video while processing; q or Escape stops it",
    )
    args = parser.parse_args()

    args.target = args.target.strip()
    if not args.target:
        parser.error("--object cannot be empty")
    cleaned_distractors = []
    seen_prompts = {args.target.casefold()}
    for distractor in args.distractor:
        distractor = distractor.strip()
        if distractor and distractor.casefold() not in seen_prompts:
            cleaned_distractors.append(distractor)
            seen_prompts.add(distractor.casefold())
    args.distractor = cleaned_distractors
    visual_prompt_enabled = (
        args.reference_image is not None or args.reference_frame is not None
    )
    if visual_prompt_enabled and not args.reference_box:
        parser.error(
            "--reference-image/--reference-frame requires at least one "
            "--reference-box"
        )
    if not visual_prompt_enabled and args.reference_box:
        parser.error(
            "--reference-box requires --reference-image or --reference-frame"
        )
    if visual_prompt_enabled and args.distractor:
        parser.error("--distractor is available only in text-prompt mode")
    if args.reference_frame is not None and args.reference_frame < 0:
        parser.error("--reference-frame must be zero or greater")
    if args.confidence is None:
        args.confidence = 0.05 if visual_prompt_enabled else 0.25
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be greater than 0 and at most 1")
    if not 0.0 < args.iou <= 1.0:
        parser.error("--iou must be greater than 0 and at most 1")
    if args.input_size <= 0:
        parser.error("--input-size must be greater than zero")
    if args.every <= 0:
        parser.error("--every must be greater than zero")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")
    if args.start_frame < 0:
        parser.error("--start-frame must be zero or greater")
    return args


def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    target: str,
    cached: bool,
) -> None:
    left, top, width, height = detection.box
    right = left + width
    bottom = top + height
    color = (0, 190, 255) if cached else (0, 255, 0)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 3)

    suffix = " (between samples)" if cached else ""
    label = f"{target}: {detection.confidence:.0%}{suffix}"
    text_y = max(35, top - 12)
    cv2.putText(
        frame,
        label,
        (left, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def load_reference_image(
    video_path: Path,
    reference_image_path: Path | None,
    reference_frame: int | None,
) -> tuple[str | np.ndarray | None, np.ndarray | None, str]:
    """Load the visual-prompt reference and return source, pixels, and label."""
    if reference_image_path is not None:
        resolved_path = reference_image_path.expanduser().resolve()
        image = cv2.imread(str(resolved_path))
        if image is None or image.size == 0:
            raise SystemExit(f"Could not read reference image: {resolved_path}")
        return str(resolved_path), image, str(resolved_path)

    if reference_frame is not None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise SystemExit(f"Could not open video for reference frame: {video_path}")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)
            ok, image = capture.read()
        finally:
            capture.release()
        if not ok or image is None or image.size == 0:
            raise SystemExit(
                f"Could not read reference frame {reference_frame} from {video_path}"
            )
        return image, image, f"video frame {reference_frame}"

    return None, None, "text prompt"


def validate_reference_boxes(
    boxes: Sequence[Sequence[float]],
    reference_pixels: np.ndarray | None,
) -> None:
    """Ensure every visual-prompt box is inside its reference image."""
    if reference_pixels is None:
        return
    height, width = reference_pixels.shape[:2]
    for box_index, (left, top, right, bottom) in enumerate(boxes, start=1):
        valid = (
            0.0 <= left < right <= width
            and 0.0 <= top < bottom <= height
        )
        if not valid:
            raise SystemExit(
                f"Reference box {box_index} must fit inside the {width}x{height} "
                f"reference image; got {left:g} {top:g} {right:g} {bottom:g}."
            )


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")
    if not model_path.is_file():
        raise SystemExit(f"YOLOE model not found: {model_path}")
    if output_path == video_path:
        raise SystemExit("--output must be different from --video")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_source, reference_pixels, reference_label = load_reference_image(
        video_path,
        args.reference_image,
        args.reference_frame,
    )
    validate_reference_boxes(args.reference_box, reference_pixels)

    try:
        detector = ObjectDetector(
            model_path,
            args.target,
            args.distractor,
            reference_source,
            args.reference_box,
            args.input_size,
            args.confidence,
            args.iou,
        )
    except Exception as error:
        raise SystemExit(f"Could not initialize YOLOE: {error}") from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise SystemExit("The video reports an invalid frame size.")
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    if expected_frames > 0 and args.start_frame >= expected_frames:
        capture.release()
        raise SystemExit(
            f"--start-frame {args.start_frame} is beyond the final frame "
            f"({expected_frames - 1})."
        )
    if args.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not create output video: {output_path}")

    print(f"Video: {video_path}")
    print(f"Target: {args.target}")
    if reference_source is not None:
        print(
            f"Prompt mode: visual; reference: {reference_label}; "
            f"example boxes: {len(args.reference_box)}"
        )
    elif args.distractor:
        print("Prompt mode: text")
        print(f"Competing classes: {', '.join(args.distractor)}")
    else:
        print("Prompt mode: text")
    print(f"Frames: {expected_frames}; detection interval: every {args.every}")
    print(f"Output: {output_path}")

    frame_count = 0
    source_frame_index = args.start_frame
    inference_count = 0
    detection_count = 0
    confidence_total = 0.0
    strongest_confidence = 0.0
    cached_detection: Detection | None = None
    started_at = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                break
            if args.max_frames is not None and frame_count >= args.max_frames:
                break

            detection_ran = frame_count % args.every == 0
            if detection_ran:
                inference_count += 1
                cached_detection = detector.detect(frame)
                if cached_detection is not None:
                    detection_count += 1
                    confidence_total += cached_detection.confidence
                    strongest_confidence = max(
                        strongest_confidence,
                        cached_detection.confidence,
                    )

            annotated_frame = frame.copy()
            if cached_detection is not None:
                draw_detection(
                    annotated_frame,
                    cached_detection,
                    args.target,
                    cached=not detection_ran,
                )

            status = (
                f"Video frame {source_frame_index + 1} | "
                f"sample detections {detection_count}/{inference_count}"
            )
            cv2.rectangle(annotated_frame, (0, 0), (width, 55), (25, 25, 25), -1)
            cv2.putText(
                annotated_frame,
                status,
                (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated_frame)
            frame_count += 1
            source_frame_index += 1

            if frame_count % max(1, round(fps * 5)) == 0:
                print(f"Processed {frame_count} frames...")

            if args.preview:
                cv2.imshow("Isolated Object Detection Video Check", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Stopped early by user.")
                    break
    finally:
        capture.release()
        writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started_at
    detection_rate = detection_count / inference_count if inference_count else 0.0
    average_confidence = (
        confidence_total / detection_count if detection_count else 0.0
    )

    print("\nResults")
    print(f"Processed frames: {frame_count}")
    print(f"Inference frames: {inference_count}")
    print(f"Frames with target: {detection_count}")
    print(f"Detection rate: {detection_rate:.1%}")
    print(f"Average confidence when detected: {average_confidence:.1%}")
    print(f"Strongest confidence: {strongest_confidence:.1%}")
    print(f"Processing time: {elapsed:.1f} seconds")
    print(f"Annotated video: {output_path}")

    if detection_count == 0:
        print("RESULT: FAIL - the target was never detected.")
        return 1

    print("RESULT: PASS - the target was detected at least once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
