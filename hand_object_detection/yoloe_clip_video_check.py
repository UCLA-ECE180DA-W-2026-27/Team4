"""Verify YOLOE video candidates with offline CLIP crop classification.

YOLOE proposes possible target boxes with a deliberately low threshold. CLIP
then compares each crop with the requested product and several competing
product descriptions. Nearby frames vote on each candidate so a single blurry
crop is not enough to verify a detection.

This is an isolated offline test. It does not import or change the live hand-
guidance application and does not upload video frames.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import chdir
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = SCRIPT_DIRECTORY.parent / "IMG_5095.MOV"
DEFAULT_MODEL_PATH = SCRIPT_DIRECTORY / "yoloe-26n-seg.pt"
DEFAULT_CLIP_DIRECTORY = SCRIPT_DIRECTORY / "weights" / "clip"
DEFAULT_OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "yoloe_clip_video_test_output"
DEFAULT_NEGATIVE_PROMPTS = (
    "a granola bar",
    "a cereal box",
    "a snack bag",
    "a jar of jam",
    "canned food",
    "a beverage bottle",
    "another grocery product",
)


class YoloECandidateDetector:
    """Return every YOLOE box assigned to the requested target class."""

    def __init__(
        self,
        model_path: Path,
        target: str,
        distractors: Sequence[str],
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
                "Install ultralytics and its CLIP text encoder in the active "
                "environment."
            ) from error

        self._input_size = input_size
        self._confidence = confidence
        self._iou = iou
        self._max_detections = max_detections
        self._model = YOLOE(str(model_path), verbose=False)
        with chdir(SCRIPT_DIRECTORY):
            self._model.set_classes([target, *distractors])

    def detect(
        self, frame: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], float]]:
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
        detections: list[tuple[tuple[int, int, int, int], float]] = []
        for coordinates_row, confidence, class_id in zip(
            coordinates, confidences, class_ids
        ):
            if int(class_id) != 0:
                continue
            left_float, top_float, right_float, bottom_float = coordinates_row
            left = max(0, min(int(math.floor(left_float)), width - 1))
            top = max(0, min(int(math.floor(top_float)), height - 1))
            right = max(left + 1, min(int(math.ceil(right_float)), width))
            bottom = max(top + 1, min(int(math.ceil(bottom_float)), height))
            detections.append(
                ((left, top, right - left, bottom - top), float(confidence))
            )
        return detections


def padded_box(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    padding: float,
) -> tuple[int, int, int, int]:
    left, top, width, height = box
    frame_height, frame_width = frame_shape[:2]
    horizontal_padding = round(width * padding)
    vertical_padding = round(height * padding)
    return (
        max(0, left - horizontal_padding),
        max(0, top - vertical_padding),
        min(frame_width, left + width + horizontal_padding),
        min(frame_height, top + height + vertical_padding),
    )


def load_frame(
    capture: cv2.VideoCapture, frame_index: int
) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        return None
    return frame


@dataclass
class CropVote:
    path: str
    frame_index: int
    offset: int
    target_probability: float = 0.0
    target_won: bool = False
    top_prompt: str = ""
    top_probability: float = 0.0


@dataclass
class Candidate:
    candidate_id: int
    box: tuple[int, int, int, int]
    yolo_confidence: float
    votes: list[CropVote] = field(default_factory=list)
    target_wins: int = 0
    average_target_probability: float = 0.0
    verified: bool = False


@dataclass
class FrameResult:
    frame_index: int
    timestamp_seconds: float
    candidates: list[Candidate]


class ClipCropVerifier:
    """Classify candidate crops against one target and competing prompts."""

    def __init__(
        self,
        target_prompts: Sequence[str],
        negative_prompts: Sequence[str],
        weights_directory: Path,
    ) -> None:
        try:
            import clip
            import torch
        except ImportError as error:
            raise RuntimeError(
                "CLIP verification requires the installed clip and torch packages"
            ) from error

        self._torch = torch
        self.target_prompt_count = len(target_prompts)
        self.prompts = [
            *(f"a photo of {target}" for target in target_prompts),
            *(f"a photo of {prompt}" for prompt in negative_prompts),
        ]
        self.model, self.preprocess = clip.load(
            "ViT-B/32",
            device="cpu",
            jit=False,
            download_root=str(weights_directory),
        )
        self.model.eval()
        tokens = clip.tokenize(self.prompts)
        with torch.inference_mode():
            text_features = self.model.encode_text(tokens)
            self.text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

    def classify(
        self,
        paths: Sequence[Path],
        batch_size: int,
    ) -> dict[str, tuple[float, bool, str, float]]:
        """Return target probability, winning prompt, and winning probability."""
        results: dict[str, tuple[float, bool, str, float]] = {}
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = self._torch.stack(
                [
                    self.preprocess(Image.open(path).convert("RGB"))
                    for path in batch_paths
                ]
            )
            with self._torch.inference_mode():
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                probabilities = (
                    100.0 * image_features @ self.text_features.T
                ).softmax(dim=-1)
            for path, row in zip(batch_paths, probabilities):
                winning_index = int(row.argmax().item())
                results[str(path.resolve())] = (
                    float(row[: self.target_prompt_count].sum().item()),
                    winning_index < self.target_prompt_count,
                    self.prompts[winning_index],
                    float(row[winning_index].item()),
                )
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test YOLOE proposals plus offline CLIP crop verification."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--object", dest="target", required=True)
    parser.add_argument(
        "--clip-target-prompt",
        action="append",
        default=None,
        help=(
            "CLIP description when different wording helps classification; "
            "repeat for equivalent products or brands"
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        action="append",
        default=None,
        help=(
            "Competing CLIP description; repeat to replace the default grocery "
            "alternatives"
        ),
    )
    parser.add_argument(
        "--detector-distractor",
        action="append",
        default=[],
        help="Competing YOLOE class; repeat for multiple detector classes",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--clip-weights-dir", type=Path, default=DEFAULT_CLIP_DIRECTORY
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Annotated MP4 path (default: OUTPUT_DIR/verified_video.mp4)",
    )
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--input-size", type=int, default=960)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, default=None)
    parser.add_argument("--sample-every-seconds", type=float, default=1.0)
    parser.add_argument(
        "--nearby-frame-offset",
        type=int,
        action="append",
        default=None,
        help="Nearby frame used for voting; repeat (default: -2, 0, 2)",
    )
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--minimum-crop-width", type=int, default=224)
    parser.add_argument("--clip-confidence", type=float, default=0.70)
    parser.add_argument("--minimum-target-votes", type=int, default=3)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    args.target = args.target.strip()
    if not args.target:
        parser.error("--object cannot be empty")
    if args.clip_target_prompt is None:
        args.clip_target_prompt = [args.target]
    else:
        args.clip_target_prompt = [
            prompt.strip()
            for prompt in args.clip_target_prompt
            if prompt.strip()
        ]
        if not args.clip_target_prompt:
            parser.error("at least one --clip-target-prompt must be non-empty")
    if args.negative_prompt is None:
        args.negative_prompt = list(DEFAULT_NEGATIVE_PROMPTS)
    else:
        args.negative_prompt = [
            prompt.strip() for prompt in args.negative_prompt if prompt.strip()
        ]
    if not args.negative_prompt:
        parser.error("at least one non-empty --negative-prompt is required")
    args.detector_distractor = [
        prompt.strip()
        for prompt in args.detector_distractor
        if prompt.strip()
    ]
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be greater than 0 and at most 1")
    if not 0.0 < args.iou <= 1.0:
        parser.error("--iou must be greater than 0 and at most 1")
    if not 0.0 < args.clip_confidence <= 1.0:
        parser.error("--clip-confidence must be greater than 0 and at most 1")
    if args.input_size <= 0 or args.max_detections <= 0:
        parser.error("--input-size and --max-detections must be positive")
    if args.start_seconds < 0.0:
        parser.error("--start-seconds cannot be negative")
    if args.end_seconds is not None and args.end_seconds < args.start_seconds:
        parser.error("--end-seconds cannot be earlier than --start-seconds")
    if args.sample_every_seconds <= 0.0:
        parser.error("--sample-every-seconds must be positive")
    if not 0.0 <= args.crop_padding <= 1.0:
        parser.error("--crop-padding must be from 0 to 1")
    if args.minimum_crop_width <= 0:
        parser.error("--minimum-crop-width must be positive")
    if args.minimum_target_votes <= 0 or args.clip_batch_size <= 0:
        parser.error("vote count and batch size must be positive")
    if args.nearby_frame_offset is None:
        args.nearby_frame_offset = [-2, 0, 2]
    else:
        args.nearby_frame_offset = sorted(
            set(args.nearby_frame_offset + [0])
        )
    return args


def save_crop(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    padding: float,
    minimum_width: int,
    output_path: Path,
) -> bool:
    left, top, right, bottom = padded_box(box, frame.shape, padding)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return False
    if crop.shape[1] < minimum_width:
        scale = minimum_width / crop.shape[1]
        crop = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    return bool(cv2.imwrite(str(output_path), crop))


def apply_clip_results(
    frame_results: Sequence[FrameResult],
    classifications: dict[str, tuple[float, bool, str, float]],
    confidence_threshold: float,
    minimum_votes: int,
) -> None:
    for frame_result in frame_results:
        for candidate in frame_result.candidates:
            for vote in candidate.votes:
                classification = classifications.get(str(Path(vote.path).resolve()))
                if classification is None:
                    continue
                (
                    vote.target_probability,
                    vote.target_won,
                    vote.top_prompt,
                    vote.top_probability,
                ) = classification
            if not candidate.votes:
                continue
            candidate.target_wins = sum(
                vote.target_won
                for vote in candidate.votes
            )
            candidate.average_target_probability = float(
                np.mean(
                    [vote.target_probability for vote in candidate.votes]
                )
            )
            required_votes = min(minimum_votes, len(candidate.votes))
            candidate.verified = (
                candidate.target_wins >= required_votes
                and candidate.average_target_probability >= confidence_threshold
            )


def annotate_frame(
    frame: np.ndarray,
    frame_result: FrameResult,
    target: str,
) -> np.ndarray:
    annotated = frame.copy()
    for candidate in frame_result.candidates:
        left, top, width, height = candidate.box
        right, bottom = left + width, top + height
        color = (0, 200, 0) if candidate.verified else (0, 165, 255)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 3)
        status = "verified" if candidate.verified else "rejected"
        label = (
            f"{target} YOLO {candidate.yolo_confidence:.0%} | {status} "
            f"CLIP {candidate.average_target_probability:.0%} "
            f"({candidate.target_wins}/{len(candidate.votes)})"
        )
        cv2.putText(
            annotated,
            label,
            (left, max(28, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        annotated,
        f"{frame_result.timestamp_seconds:.2f}s",
        (18, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return annotated


def write_verified_video(
    video_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    frame_results: Sequence[FrameResult],
    target: str,
) -> None:
    """Write a smooth source clip with the nearest verified sample overlaid."""
    if not frame_results:
        raise RuntimeError("No sampled frames are available for video output")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    sample_position = 0
    try:
        for frame_index in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                break
            while (
                sample_position + 1 < len(frame_results)
                and abs(
                    frame_results[sample_position + 1].frame_index
                    - frame_index
                )
                <= abs(
                    frame_results[sample_position].frame_index - frame_index
                )
            ):
                sample_position += 1

            sample = frame_results[sample_position]
            verified = [
                candidate
                for candidate in sample.candidates
                if candidate.verified
            ]
            for candidate in verified:
                left, top, box_width, box_height = candidate.box
                right = left + box_width
                bottom = top + box_height
                cv2.rectangle(
                    frame, (left, top), (right, bottom), (0, 220, 0), 5
                )
                label = (
                    f"VERIFIED: {target} | "
                    f"CLIP {candidate.average_target_probability:.0%}"
                )
                cv2.putText(
                    frame,
                    label,
                    (left, max(32, top - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 220, 0),
                    3,
                    cv2.LINE_AA,
                )

            status = (
                f"Target verified ({len(verified)})"
                if verified
                else "Target not verified"
            )
            status_color = (0, 220, 0) if verified else (0, 165, 255)
            cv2.rectangle(frame, (0, 0), (width, 70), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"{frame_index / fps:.2f}s | {status}",
                (18, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                3,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        capture.release()
        writer.release()


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    clip_weights_directory = args.clip_weights_dir.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()
    output_video_path = (
        (output_directory / "verified_video.mp4")
        if args.output_video is None
        else args.output_video.expanduser().resolve()
    )
    crops_directory = output_directory / "crops"
    frames_directory = output_directory / "frames"

    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")
    if not model_path.is_file():
        raise SystemExit(f"YOLOE model not found: {model_path}")
    crops_directory.mkdir(parents=True, exist_ok=True)
    frames_directory.mkdir(parents=True, exist_ok=True)
    clip_weights_directory.mkdir(parents=True, exist_ok=True)

    try:
        detector = YoloECandidateDetector(
            model_path=model_path,
            target=args.target,
            distractors=args.detector_distractor,
            input_size=args.input_size,
            confidence=args.confidence,
            iou=args.iou,
            max_detections=args.max_detections,
        )
        verifier = ClipCropVerifier(
            target_prompts=args.clip_target_prompt,
            negative_prompts=args.negative_prompt,
            weights_directory=clip_weights_directory,
        )
    except Exception as error:
        raise SystemExit(f"Could not initialize models: {error}") from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    final_frame = max(0, frame_count - 1)
    start_frame = min(final_frame, round(args.start_seconds * fps))
    end_frame = (
        final_frame
        if args.end_seconds is None
        else min(final_frame, round(args.end_seconds * fps))
    )
    step = max(1, round(args.sample_every_seconds * fps))

    print(f"Video: {video_path}")
    print(f"Visual target: {args.target}")
    print(f"CLIP target descriptions: {', '.join(args.clip_target_prompt)}")
    print(f"CLIP alternatives: {', '.join(args.negative_prompt)}")
    print(
        f"Sampling frames {start_frame} through {end_frame} every {step} frames"
    )

    frame_results: list[FrameResult] = []
    source_frames: dict[int, np.ndarray] = {}
    crop_paths: list[Path] = []
    candidate_id = 0
    started_at = time.perf_counter()

    try:
        for frame_index in range(start_frame, end_frame + 1, step):
            frame = load_frame(capture, frame_index)
            if frame is None:
                print(f"Warning: could not read frame {frame_index}")
                continue
            source_frames[frame_index] = frame
            detections = detector.detect(frame)
            candidates: list[Candidate] = []
            for box, confidence in detections:
                candidate_id += 1
                votes: list[CropVote] = []
                for offset in args.nearby_frame_offset:
                    nearby_index = min(final_frame, max(0, frame_index + offset))
                    nearby_frame = (
                        frame
                        if nearby_index == frame_index
                        else load_frame(capture, nearby_index)
                    )
                    if nearby_frame is None:
                        continue
                    crop_path = crops_directory / (
                        f"candidate_{candidate_id:05d}_frame_{nearby_index:06d}"
                        f"_offset_{offset:+d}.jpg"
                    )
                    if save_crop(
                        nearby_frame,
                        box,
                        args.crop_padding,
                        args.minimum_crop_width,
                        crop_path,
                    ):
                        votes.append(
                            CropVote(
                                path=str(crop_path),
                                frame_index=nearby_index,
                                offset=offset,
                            )
                        )
                        crop_paths.append(crop_path)
                candidates.append(
                    Candidate(
                        candidate_id=candidate_id,
                        box=box,
                        yolo_confidence=confidence,
                        votes=votes,
                    )
                )
            frame_results.append(
                FrameResult(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    candidates=candidates,
                )
            )
            print(
                f"{frame_index / fps:6.2f}s: "
                f"{len(candidates)} YOLOE candidate(s)"
            )
    finally:
        capture.release()

    print(f"Classifying {len(crop_paths)} nearby-frame crops with CLIP...")
    classifications = verifier.classify(crop_paths, args.clip_batch_size)
    apply_clip_results(
        frame_results,
        classifications,
        args.clip_confidence,
        args.minimum_target_votes,
    )

    verified_frames = 0
    verified_candidates = 0
    total_candidates = 0
    for frame_result in frame_results:
        total_candidates += len(frame_result.candidates)
        frame_verified = sum(
            candidate.verified for candidate in frame_result.candidates
        )
        verified_candidates += frame_verified
        verified_frames += bool(frame_verified)
        annotated = annotate_frame(
            source_frames[frame_result.frame_index], frame_result, args.target
        )
        frame_path = frames_directory / (
            f"frame_{frame_result.frame_index:06d}_"
            f"{frame_result.timestamp_seconds:07.2f}s.jpg"
        )
        cv2.imwrite(str(frame_path), annotated)
        print(
            f"{frame_result.timestamp_seconds:6.2f}s: "
            f"verified {frame_verified}/{len(frame_result.candidates)}"
        )
        if args.preview:
            cv2.imshow("YOLOE + CLIP Video Check", annotated)
            if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                break
    if args.preview:
        cv2.destroyAllWindows()

    try:
        write_verified_video(
            video_path=video_path,
            output_path=output_video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            frame_results=frame_results,
            target=args.target,
        )
    except Exception as error:
        raise SystemExit(f"Could not create annotated video: {error}") from error

    report = {
        "video": str(video_path),
        "target": args.target,
        "clip_target_prompts": args.clip_target_prompt,
        "clip_prompts": verifier.prompts,
        "clip_confidence": args.clip_confidence,
        "minimum_target_votes": args.minimum_target_votes,
        "frames": [asdict(result) for result in frame_results],
    }
    results_path = output_directory / "results.json"
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started_at

    print("\nResults")
    print(f"Sampled frames: {len(frame_results)}")
    print(f"YOLOE candidates: {total_candidates}")
    print(f"CLIP-verified candidates: {verified_candidates}")
    print(f"Frames with a verified target: {verified_frames}")
    print(f"Processing time: {elapsed:.1f} seconds")
    print(f"Annotated frames: {frames_directory}")
    print(f"Annotated video: {output_video_path}")
    print(f"Detailed report: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
