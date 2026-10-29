"""Detect a palm and a selected object in estimated camera-space XYZ.

Coordinate convention follows OpenCV's camera frame:

* origin: camera optical center (the lens)
* +X: camera-left
* +Y: camera-up
* +Z: forward from the camera into the scene

MediaPipe tracks the palm. YOLOE-26 detects a selected object from a text
prompt and draws its bounding box. The Logitech C270 is a monocular RGB camera,
not a depth camera, so both positions are estimates rather than direct or
safety-grade distance measurements.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


DEFAULT_MODEL_PATH = Path(__file__).resolve().with_name("hand_landmarker.task")
DEFAULT_OBJECT_MODEL_PATH = Path(__file__).resolve().with_name(
    "yoloe-26n-seg.pt"
)
MAX_HANDS = 4
DEFAULT_MIN_HAND_SIZE = 0.10
DEFAULT_DIAGONAL_FOV_DEGREES = 55.0
DEFAULT_OBJECT_CONFIDENCE = 0.25
DEFAULT_OBJECT_NMS_THRESHOLD = 0.45
DEFAULT_OBJECT_INPUT_SIZE = 320
DEFAULT_OBJECT_DETECTION_INTERVAL = 3
DEFAULT_GUIDANCE_RADIUS_FRACTION = 0.10
MAX_CAMERA_READ_FAILURES = 30
CAMERA_RETRY_DELAY_SECONDS = 0.1
LAPTOP_LOGITECH_CAMERA_INDEX = 0
PI_LOGITECH_CAMERA_INDEX = 0
LOGITECH_FRAME_WIDTH = 1280
LOGITECH_FRAME_HEIGHT = 720
LOGITECH_FRAME_RATE = 30

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
PALM_IDS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

PALM_CONNECTIONS = (
    (WRIST, INDEX_MCP),
    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, PINKY_MCP),
    (PINKY_MCP, WRIST),
)


class VectorSmoother:
    """Apply exponential smoothing to a three-dimensional position."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._value: np.ndarray | None = None

    def update(self, value: np.ndarray) -> np.ndarray:
        if self._value is None:
            self._value = value.astype(np.float64, copy=True)
        else:
            self._value = (
                (1.0 - self.alpha) * self._value
                + self.alpha * value
            )
        return self._value.copy()

    def reset(self) -> None:
        self._value = None


@dataclass(frozen=True)
class ObjectDetection:
    """One object detection in raw-camera pixel coordinates."""

    box: tuple[int, int, int, int]
    confidence: float


class YoloE26ObjectDetector:
    """Run YOLOE-26 with one open-vocabulary text prompt."""

    def __init__(
        self,
        model_path: Path,
        target_name: str,
        input_size: int,
        confidence_threshold: float,
        nms_threshold: float,
    ) -> None:
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        try:
            import clip  # noqa: F401
            from ultralytics import YOLOE
        except ImportError as error:
            raise RuntimeError(
                "YOLOE-26 requires ultralytics and the Ultralytics CLIP "
                "text encoder. See the README setup instructions."
            ) from error

        self.model = YOLOE(str(model_path), verbose=False)
        self.model.set_classes([target_name])

    def detect(self, frame: np.ndarray) -> ObjectDetection | None:
        """Return the strongest bounding box matching the text prompt."""
        frame_height, frame_width = frame.shape[:2]
        results = self.model.predict(
            frame,
            imgsz=self.input_size,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            device="cpu",
            max_det=20,
            verbose=False,
        )
        if not results:
            return None
        result_boxes = results[0].boxes
        if result_boxes is None or len(result_boxes) == 0:
            return None

        confidences = result_boxes.conf.detach().cpu().numpy()
        coordinates = result_boxes.xyxy.detach().cpu().numpy()
        best_index = int(np.argmax(confidences))
        left_float, top_float, right_float, bottom_float = coordinates[best_index]
        left = max(0, min(int(math.floor(left_float)), frame_width - 1))
        top = max(0, min(int(math.floor(top_float)), frame_height - 1))
        right = max(left + 1, min(int(math.ceil(right_float)), frame_width))
        bottom = max(top + 1, min(int(math.ceil(bottom_float)), frame_height))
        return ObjectDetection(
            box=(left, top, right - left, bottom - top),
            confidence=float(confidences[best_index]),
        )


def palm_size_fraction(
    landmarks: Sequence,
    frame_width: int,
    frame_height: int,
) -> float:
    """Return the largest visible palm span, normalized by frame size."""
    palm_points = [
        (landmarks[index].x * frame_width, landmarks[index].y * frame_height)
        for index in PALM_IDS
    ]
    largest_span = max(
        math.hypot(
            palm_points[first][0] - palm_points[second][0],
            palm_points[first][1] - palm_points[second][1],
        )
        for first in range(len(palm_points))
        for second in range(first + 1, len(palm_points))
    )
    return largest_span / min(frame_width, frame_height)


def select_foreground_hand(
    hands: Sequence[Sequence],
    frame_width: int,
    frame_height: int,
    min_hand_size: float,
) -> tuple[int | None, float]:
    """Return the largest qualifying hand index and its palm-size fraction."""
    if not hands:
        return None, 0.0

    sizes = [
        palm_size_fraction(hand, frame_width, frame_height)
        for hand in hands
    ]
    selected_index = int(np.argmax(sizes))
    selected_size = sizes[selected_index]
    if selected_size < min_hand_size:
        return None, selected_size
    return selected_index, selected_size


def palm_center_normalized(image_landmarks: Sequence) -> np.ndarray:
    """Return the 2-D palm center as normalized image coordinates."""
    return np.mean(
        np.array(
            [
                (image_landmarks[index].x, image_landmarks[index].y)
                for index in PALM_IDS
            ],
            dtype=np.float64,
        ),
        axis=0,
    )


def camera_matrix_from_diagonal_fov(
    frame_width: int,
    frame_height: int,
    diagonal_fov_degrees: float,
) -> np.ndarray:
    """Approximate camera intrinsics using a centered square-pixel model."""
    diagonal_pixels = math.hypot(frame_width, frame_height)
    focal_length_pixels = diagonal_pixels / (
        2.0 * math.tan(math.radians(diagonal_fov_degrees) / 2.0)
    )
    return np.array(
        (
            (focal_length_pixels, 0.0, (frame_width - 1) / 2.0),
            (0.0, focal_length_pixels, (frame_height - 1) / 2.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def estimate_palm_camera_xyz(
    image_landmarks: Sequence,
    world_landmarks: Sequence,
    frame_width: int,
    frame_height: int,
    diagonal_fov_degrees: float,
) -> np.ndarray | None:
    """Estimate palm-center XYZ in the camera frame, in meters.

    MediaPipe's world landmarks provide an approximate metric hand shape whose
    origin is local to the hand. solvePnP estimates that hand's rigid transform
    into the camera frame. The transformed centroid of the wrist and four MCP
    joints is used as the palm center.
    """
    if len(image_landmarks) != len(world_landmarks) or len(image_landmarks) < 6:
        return None

    object_points = np.array(
        [(point.x, point.y, point.z) for point in world_landmarks],
        dtype=np.float64,
    )
    image_points = np.array(
        [
            (
                point.x * (frame_width - 1),
                point.y * (frame_height - 1),
            )
            for point in image_landmarks
        ],
        dtype=np.float64,
    )
    camera_matrix = camera_matrix_from_diagonal_fov(
        frame_width,
        frame_height,
        diagonal_fov_degrees,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            return None

        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            rotation_vector,
            translation_vector,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
    except cv2.error:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    palm_center_in_hand_frame = np.mean(object_points[list(PALM_IDS)], axis=0)
    palm_center_in_camera_frame = (
        rotation_matrix @ palm_center_in_hand_frame
        + translation_vector.reshape(3)
    )

    if not np.all(np.isfinite(palm_center_in_camera_frame)):
        return None
    if palm_center_in_camera_frame[2] <= 0.0:
        return None
    # Convert OpenCV's +right/+down image axes to the requested
    # +left/+up axes while retaining +Z forward.
    palm_center_in_camera_frame[0] *= -1.0
    palm_center_in_camera_frame[1] *= -1.0
    return palm_center_in_camera_frame


def estimate_object_camera_xyz(
    box: tuple[int, int, int, int],
    physical_width_meters: float,
    frame_width: int,
    frame_height: int,
    diagonal_fov_degrees: float,
) -> np.ndarray | None:
    """Estimate a bounding-box center in camera XYZ using known object width."""
    left, top, box_width, box_height = box
    if box_width <= 0 or box_height <= 0 or physical_width_meters <= 0.0:
        return None

    camera_matrix = camera_matrix_from_diagonal_fov(
        frame_width,
        frame_height,
        diagonal_fov_degrees,
    )
    focal_length_x = float(camera_matrix[0, 0])
    focal_length_y = float(camera_matrix[1, 1])
    principal_x = float(camera_matrix[0, 2])
    principal_y = float(camera_matrix[1, 2])
    center_x = left + box_width / 2.0
    center_y = top + box_height / 2.0

    depth = focal_length_x * physical_width_meters / box_width
    opencv_x = (center_x - principal_x) * depth / focal_length_x
    opencv_y = (center_y - principal_y) * depth / focal_length_y

    # Flip OpenCV's right/down axes to the requested left/up axes.
    return np.array((-opencv_x, -opencv_y, depth), dtype=np.float64)


def resolve_target(
    requested_name: str | None,
    requested_width_cm: float | None,
) -> tuple[str, float | None]:
    """Prompt for the target and optional physical width before capture."""
    if requested_name is None:
        try:
            requested_name = input("Object to detect: ").strip()
        except EOFError as error:
            raise SystemExit("Pass the target with --object NAME.") from error

    target_name = requested_name.strip()
    if not target_name:
        raise SystemExit("Object description cannot be empty.")

    if requested_width_cm is not None and requested_width_cm <= 0.0:
        raise SystemExit("Object width must be greater than zero centimeters.")

    width_meters = (
        None if requested_width_cm is None else requested_width_cm / 100.0
    )
    return target_name, width_meters


def _display_pixel(
    landmark,
    width: int,
    height: int,
    mirrored: bool,
) -> tuple[int, int]:
    x = int(round(landmark.x * (width - 1)))
    y = int(round(landmark.y * (height - 1)))
    if mirrored:
        x = width - 1 - x
    return x, y


def _display_normalized_point(
    normalized_point: np.ndarray,
    width: int,
    height: int,
    mirrored: bool,
) -> tuple[int, int]:
    x = int(round(float(normalized_point[0]) * (width - 1)))
    y = int(round(float(normalized_point[1]) * (height - 1)))
    if mirrored:
        x = width - 1 - x
    return x, y


def _display_object_center(
    detection: ObjectDetection,
    frame_width: int,
    mirrored: bool,
) -> tuple[int, int]:
    """Return the detected object's center in displayed-frame pixels."""
    left, top, box_width, box_height = detection.box
    if mirrored:
        left = frame_width - left - box_width
    return left + box_width // 2, top + box_height // 2


def guidance_text(
    palm_center: tuple[int, int],
    object_center: tuple[int, int],
    alignment_radius: int,
) -> tuple[str, bool]:
    """Return displayed-frame movement guidance and alignment state."""
    delta_x = object_center[0] - palm_center[0]
    delta_y = object_center[1] - palm_center[1]
    if math.hypot(delta_x, delta_y) <= alignment_radius:
        return "ALIGNED: MOVE FORWARD / REACH", True

    directions: list[str] = []
    axis_deadband = max(1.0, alignment_radius * 0.35)
    if delta_x < -axis_deadband:
        directions.append("LEFT")
    elif delta_x > axis_deadband:
        directions.append("RIGHT")
    if delta_y < -axis_deadband:
        directions.append("UP")
    elif delta_y > axis_deadband:
        directions.append("DOWN")

    if not directions:
        directions.append("LEFT" if delta_x < 0 else "RIGHT")
    return "MOVE HAND: " + " | ".join(directions), False


def draw_hand_and_palm_center(
    frame: np.ndarray,
    landmarks: Sequence,
    palm_center: np.ndarray,
    mirrored: bool,
) -> None:
    """Draw the selected hand and emphasize its estimated palm center."""
    height, width = frame.shape[:2]

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            _display_pixel(landmarks[start], width, height, mirrored),
            _display_pixel(landmarks[end], width, height, mirrored),
            (170, 170, 170),
            2,
            cv2.LINE_AA,
        )

    for start, end in PALM_CONNECTIONS:
        cv2.line(
            frame,
            _display_pixel(landmarks[start], width, height, mirrored),
            _display_pixel(landmarks[end], width, height, mirrored),
            (0, 220, 255),
            4,
            cv2.LINE_AA,
        )

    for landmark in landmarks:
        cv2.circle(
            frame,
            _display_pixel(landmark, width, height, mirrored),
            4,
            (240, 240, 240),
            -1,
            cv2.LINE_AA,
        )

    optical_center = ((width - 1) // 2, (height - 1) // 2)
    center_pixel = _display_normalized_point(
        palm_center,
        width,
        height,
        mirrored,
    )
    cv2.line(
        frame,
        optical_center,
        center_pixel,
        (255, 190, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(frame, center_pixel, 12, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, center_pixel, 17, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "PALM CENTER",
        (center_pixel[0] + 16, center_pixel[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_object_box(
    frame: np.ndarray,
    detection: ObjectDetection,
    target_name: str,
    object_xyz: np.ndarray | None,
    mirrored: bool,
    cached: bool = False,
) -> None:
    """Draw the selected object's OpenCV bounding box and estimated center."""
    frame_height, frame_width = frame.shape[:2]
    left, top, box_width, box_height = detection.box
    if mirrored:
        left = frame_width - left - box_width

    right = min(frame_width - 1, left + box_width)
    bottom = min(frame_height - 1, top + box_height)
    center = _display_object_center(detection, frame_width, mirrored)
    color = (0, 220, 255) if cached else (80, 255, 80)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 3, cv2.LINE_AA)
    cv2.circle(frame, center, 9, color, -1, cv2.LINE_AA)
    label = f"{target_name} {detection.confidence:.0%}"
    if cached:
        label += " (last known)"
    if object_xyz is not None:
        label += (
            f"  ({object_xyz[0]:+.2f}, {object_xyz[1]:+.2f}, "
            f"{object_xyz[2]:+.2f}) m"
        )
    label_y = max(24, top - 10)
    cv2.putText(
        frame,
        label,
        (max(4, left), label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_optical_center(frame: np.ndarray) -> None:
    """Draw the image principal point assumed by the camera model."""
    height, width = frame.shape[:2]
    center_x = (width - 1) // 2
    center_y = (height - 1) // 2
    cv2.line(
        frame,
        (center_x - 12, center_y),
        (center_x + 12, center_y),
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (center_x, center_y - 12),
        (center_x, center_y + 12),
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


def _draw_text(
    frame: np.ndarray,
    text: str,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.72,
) -> None:
    cv2.putText(
        frame,
        text,
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect a palm and selected object and estimate both XYZ positions."
        )
    )
    computer_group = parser.add_mutually_exclusive_group()
    computer_group.add_argument(
        "--laptop",
        dest="computer",
        action="store_const",
        const="laptop",
        help="Use the USB Logitech camera on the laptop (default)",
    )
    computer_group.add_argument(
        "--pi",
        dest="computer",
        action="store_const",
        const="pi",
        help="Use the USB Logitech camera on a Raspberry Pi",
    )
    parser.set_defaults(computer="laptop")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Override the default Logitech camera index",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe hand_landmarker.task model",
    )
    parser.add_argument(
        "--object-model",
        type=Path,
        default=DEFAULT_OBJECT_MODEL_PATH,
        help="Path to a YOLOE-26 segmentation .pt checkpoint",
    )
    parser.add_argument(
        "--object",
        dest="object_name",
        default=None,
        help="Open-vocabulary target description; prompted when omitted",
    )
    parser.add_argument(
        "--object-width-cm",
        type=float,
        default=None,
        help="Optional target width in cm, needed only for object XYZ",
    )
    parser.add_argument(
        "--object-confidence",
        type=float,
        default=DEFAULT_OBJECT_CONFIDENCE,
        help="Minimum object confidence from 0 to 1 (default: 0.25)",
    )
    parser.add_argument(
        "--object-nms-threshold",
        type=float,
        default=DEFAULT_OBJECT_NMS_THRESHOLD,
        help="Object overlap suppression threshold from 0 to 1 (default: 0.45)",
    )
    parser.add_argument(
        "--object-input-size",
        type=int,
        default=DEFAULT_OBJECT_INPUT_SIZE,
        help="Square YOLOE-26 input size in pixels (default: 320)",
    )
    parser.add_argument(
        "--object-every",
        type=int,
        default=DEFAULT_OBJECT_DETECTION_INTERVAL,
        help="Run object detection every N camera frames (default: 3)",
    )
    parser.add_argument(
        "--guidance-radius",
        type=float,
        default=DEFAULT_GUIDANCE_RADIUS_FRACTION,
        help=(
            "Alignment radius as a fraction of the shorter frame side "
            "(default: 0.10)"
        ),
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.25,
        help="XYZ smoothing from 0 (slow) to 1 (none); default: 0.25",
    )
    parser.add_argument(
        "--min-hand-size",
        type=float,
        default=DEFAULT_MIN_HAND_SIZE,
        help=(
            "Minimum largest palm span as a fraction of the frame "
            "(default: 0.10)"
        ),
    )
    parser.add_argument(
        "--diagonal-fov",
        type=float,
        default=DEFAULT_DIAGONAL_FOV_DEGREES,
        help="Camera diagonal field of view in degrees (C270 default: 55)",
    )
    parser.add_argument(
        "--print-xyz",
        action="store_true",
        help="Print both XYZ values and separation when a width is supplied",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Display the raw camera image instead of a mirror image",
    )
    args = parser.parse_args()

    if not 0.0 < args.smoothing <= 1.0:
        parser.error("--smoothing must be greater than 0 and at most 1")
    if not 0.0 <= args.min_hand_size <= 1.0:
        parser.error("--min-hand-size must be between 0 and 1")
    if not 1.0 < args.diagonal_fov < 179.0:
        parser.error("--diagonal-fov must be between 1 and 179 degrees")
    if args.object_width_cm is not None and args.object_width_cm <= 0.0:
        parser.error("--object-width-cm must be greater than zero")
    if not 0.0 < args.object_confidence <= 1.0:
        parser.error("--object-confidence must be greater than 0 and at most 1")
    if not 0.0 < args.object_nms_threshold <= 1.0:
        parser.error("--object-nms-threshold must be greater than 0 and at most 1")
    if args.object_input_size <= 0:
        parser.error("--object-input-size must be greater than zero")
    if args.object_every <= 0:
        parser.error("--object-every must be greater than zero")
    if not 0.0 < args.guidance_radius <= 0.5:
        parser.error("--guidance-radius must be greater than 0 and at most 0.5")
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be zero or greater")
    return args


def main() -> None:
    args = parse_args()
    hand_model_path = args.model.expanduser().resolve()
    object_model_path = args.object_model.expanduser().resolve()
    if not hand_model_path.is_file():
        raise SystemExit(
            f"MediaPipe model not found: {hand_model_path}\n"
            "Place hand_landmarker.task beside this script or pass --model PATH."
        )
    if not object_model_path.is_file():
        raise SystemExit(
            f"Object model not found: {object_model_path}\n"
            "Place yoloe-26n-seg.pt beside this script or pass "
            "--object-model PATH."
        )

    target_name, target_width_meters = resolve_target(
        args.object_name,
        args.object_width_cm,
    )

    try:
        object_detector = YoloE26ObjectDetector(
            object_model_path,
            target_name,
            args.object_input_size,
            args.object_confidence,
            args.object_nms_threshold,
        )
    except Exception as error:
        raise SystemExit(
            f"Could not initialize YOLOE-26: {object_model_path}\n{error}"
        ) from error

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if args.computer == "pi":
        camera_index = (
            PI_LOGITECH_CAMERA_INDEX if args.camera is None else args.camera
        )
        camera_backend = cv2.CAP_V4L2
        backend_name = "V4L2"
    else:
        camera_index = (
            LAPTOP_LOGITECH_CAMERA_INDEX if args.camera is None else args.camera
        )
        camera_backend = cv2.CAP_AVFOUNDATION
        backend_name = "AVFoundation"

    capture = cv2.VideoCapture(camera_index, camera_backend)
    if not capture.isOpened():
        raise SystemExit(
            f"Could not open the Logitech camera at index {camera_index} "
            f"using {backend_name}.\n"
            "Check the USB connection or override the index with --camera N."
        )

    if args.computer == "pi":
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, LOGITECH_FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, LOGITECH_FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, LOGITECH_FRAME_RATE)
    print(
        f"Using Logitech camera index {camera_index} on {args.computer} "
        f"via {backend_name}."
    )
    if target_width_meters is None:
        print(f"Target: {target_name}; object XYZ disabled (no width supplied).")
    else:
        print(
            f"Target: {target_name}; known width: "
            f"{target_width_meters * 100.0:.1f} cm."
        )
    print("XYZ convention: +X left, +Y up, +Z forward; units are meters.")
    print("Depth is estimated from a monocular camera and is not safety-grade.")
    if args.print_xyz:
        print(
            "timestamp_ms,palm_x,palm_y,palm_z,object_x,object_y,object_z,"
            "distance_m"
        )

    palm_position_smoother = VectorSmoother(args.smoothing)
    object_position_smoother = VectorSmoother(args.smoothing)
    previous_timestamp_ms = -1
    window_name = f"Palm and Object Detection - {target_name}"
    consecutive_camera_failures = 0
    mirrored_display = not args.no_mirror
    frame_number = 0
    object_detection: ObjectDetection | None = None
    object_visible = False

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, camera_frame = capture.read()
                if not ok or camera_frame is None or camera_frame.size == 0:
                    consecutive_camera_failures += 1
                    if consecutive_camera_failures >= MAX_CAMERA_READ_FAILURES:
                        raise SystemExit(
                            f"Camera {camera_index} opened but did not return video frames.\n"
                            "Close other apps using the camera, allow camera access "
                            "for your terminal or editor in system settings, or try "
                            "another camera index such as --camera 1."
                        )
                    time.sleep(CAMERA_RETRY_DELAY_SECONDS)
                    continue

                consecutive_camera_failures = 0
                height, width = camera_frame.shape[:2]

                # Inference always uses the raw image so XYZ remains in the physical
                # camera frame even when the preview is mirrored for the user.
                rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = time.monotonic_ns() // 1_000_000
                timestamp_ms = max(timestamp_ms, previous_timestamp_ms + 1)
                previous_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if frame_number % args.object_every == 0:
                    current_object_detection = object_detector.detect(camera_frame)
                    object_visible = current_object_detection is not None
                    if current_object_detection is not None:
                        object_detection = current_object_detection
                frame_number += 1

                if mirrored_display:
                    display_frame = cv2.flip(camera_frame, 1)
                else:
                    display_frame = camera_frame.copy()

                cv2.rectangle(display_frame, (0, 0), (width, 220), (25, 25, 25), -1)
                draw_optical_center(display_frame)

                selected_index, selected_size = select_foreground_hand(
                    result.hand_landmarks,
                    width,
                    height,
                    args.min_hand_size,
                )

                palm_xyz: np.ndarray | None = None
                palm_display_center: tuple[int, int] | None = None
                palm_status = "PALM: not detected"
                if selected_index is not None:
                    image_landmarks = result.hand_landmarks[selected_index]
                    palm_center = palm_center_normalized(image_landmarks)
                    palm_display_center = _display_normalized_point(
                        palm_center,
                        width,
                        height,
                        mirrored_display,
                    )
                    draw_hand_and_palm_center(
                        display_frame,
                        image_landmarks,
                        palm_center,
                        mirrored_display,
                    )

                    world_landmarks_available = (
                        selected_index < len(result.hand_world_landmarks)
                    )
                    estimated_xyz = None
                    if world_landmarks_available:
                        estimated_xyz = estimate_palm_camera_xyz(
                            image_landmarks,
                            result.hand_world_landmarks[selected_index],
                            width,
                            height,
                            args.diagonal_fov,
                        )

                    if estimated_xyz is not None:
                        palm_xyz = palm_position_smoother.update(estimated_xyz)
                        palm_status = (
                            f"PALM XYZ: {palm_xyz[0]:+.2f}, "
                            f"{palm_xyz[1]:+.2f}, {palm_xyz[2]:+.2f} m"
                        )
                    else:
                        palm_position_smoother.reset()
                        palm_status = "PALM: found, but XYZ unavailable"
                elif result.hand_landmarks:
                    palm_position_smoother.reset()
                    palm_status = (
                        f"PALM: too distant ({selected_size * 100:.0f}% / "
                        f"{args.min_hand_size * 100:.0f}% minimum)"
                    )
                else:
                    palm_position_smoother.reset()

                object_xyz: np.ndarray | None = None
                object_display_center: tuple[int, int] | None = None
                object_status = f"OBJECT ({target_name}): not detected"
                if object_detection is not None:
                    object_display_center = _display_object_center(
                        object_detection,
                        width,
                        mirrored_display,
                    )
                    if target_width_meters is None:
                        object_position_smoother.reset()
                        if object_visible:
                            object_status = (
                                "OBJECT DETECTED "
                                f"({object_detection.confidence:.0%}); "
                                "XYZ unavailable without width"
                            )
                        else:
                            object_status = (
                                "OBJECT OCCLUDED: using last known center"
                            )
                    else:
                        estimated_object_xyz = estimate_object_camera_xyz(
                            object_detection.box,
                            target_width_meters,
                            width,
                            height,
                            args.diagonal_fov,
                        )
                        if estimated_object_xyz is not None:
                            object_xyz = object_position_smoother.update(
                                estimated_object_xyz
                            )
                            if object_visible:
                                object_status = (
                                    f"OBJECT XYZ: {object_xyz[0]:+.2f}, "
                                    f"{object_xyz[1]:+.2f}, "
                                    f"{object_xyz[2]:+.2f} m "
                                    f"({object_detection.confidence:.0%})"
                                )
                            else:
                                object_status = (
                                    "OBJECT OCCLUDED: using last known XYZ"
                                )
                        else:
                            object_position_smoother.reset()
                            object_status = (
                                f"OBJECT ({target_name}): XYZ unavailable"
                            )
                    draw_object_box(
                        display_frame,
                        object_detection,
                        target_name,
                        object_xyz,
                        mirrored_display,
                        cached=not object_visible,
                    )
                else:
                    object_position_smoother.reset()

                target_label = f"TARGET: {target_name}"
                if target_width_meters is not None:
                    target_label += f" ({target_width_meters * 100.0:.1f} cm wide)"
                _draw_text(
                    display_frame,
                    target_label,
                    28,
                    (255, 255, 255),
                    0.66,
                )
                _draw_text(
                    display_frame,
                    palm_status,
                    60,
                    (80, 255, 80) if palm_xyz is not None else (0, 165, 255),
                    0.62,
                )
                _draw_text(
                    display_frame,
                    object_status,
                    92,
                    (
                        (80, 255, 80)
                        if object_visible
                        else (
                            (0, 220, 255)
                            if object_detection is not None
                            else (0, 165, 255)
                        )
                    ),
                    0.62,
                )

                alignment_radius_pixels = max(
                    1,
                    int(round(args.guidance_radius * min(width, height))),
                )
                guidance = ""
                aligned = False
                if palm_display_center is not None and object_display_center is not None:
                    guidance, aligned = guidance_text(
                        palm_display_center,
                        object_display_center,
                        alignment_radius_pixels,
                    )
                elif palm_display_center is None and object_display_center is None:
                    guidance = "GUIDANCE WAITING: show palm and target object"
                elif palm_display_center is None:
                    guidance = "GUIDANCE WAITING: show palm"
                else:
                    guidance = "GUIDANCE WAITING: show target object"

                if object_display_center is not None:
                    cv2.circle(
                        display_frame,
                        object_display_center,
                        alignment_radius_pixels,
                        (80, 255, 80) if aligned else (0, 220, 255),
                        3,
                        cv2.LINE_AA,
                    )
                _draw_text(
                    display_frame,
                    guidance,
                    132,
                    (80, 255, 80) if aligned else (0, 220, 255),
                    0.72,
                )

                if target_width_meters is None:
                    _draw_text(
                        display_frame,
                        "OBJECT XYZ/DISTANCE: disabled because width was skipped",
                        170,
                        (225, 225, 225),
                        0.58,
                    )
                elif palm_xyz is not None and object_xyz is not None:
                    distance = float(np.linalg.norm(object_xyz - palm_xyz))
                    _draw_text(
                        display_frame,
                        f"PALM-OBJECT DISTANCE: {distance:.2f} m",
                        170,
                        (225, 225, 225),
                        0.58,
                    )
                    if args.print_xyz:
                        print(
                            f"{timestamp_ms},"
                            f"{palm_xyz[0]:+.6f},{palm_xyz[1]:+.6f},"
                            f"{palm_xyz[2]:+.6f},"
                            f"{object_xyz[0]:+.6f},{object_xyz[1]:+.6f},"
                            f"{object_xyz[2]:+.6f},{distance:.6f}",
                            flush=True,
                        )
                else:
                    _draw_text(
                        display_frame,
                        "PALM-OBJECT DISTANCE: waiting for both XYZ estimates",
                        170,
                        (0, 165, 255),
                        0.58,
                    )

                _draw_text(
                    display_frame,
                    "Axes: +X left, +Y up, +Z forward; origin at camera lens",
                    204,
                    (210, 210, 210),
                    0.54,
                )

                cv2.putText(
                    display_frame,
                    "Approximate monocular XYZ - press Q or Esc to quit",
                    (18, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
