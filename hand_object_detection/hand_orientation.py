"""Detect a palm and a selected object in estimated camera-space XYZ.

Coordinate convention follows OpenCV's camera frame:

* origin: camera optical center (the lens)
* +X: camera-left
* +Y: camera-up
* +Z: forward from the camera into the scene

MediaPipe tracks the palm. By default, local Qwen 4B grounds a generic object
from a text prompt and SigLIP verifies the proposed crop; YOLOE-26 remains an
optional fallback. In remote mode, the Pi retains the camera feed for hand
tracking while sending JPEG frames to a computer that returns a normalized
object box. The Logitech C270 is a monocular RGB camera, not a depth camera, so
both positions are estimates rather than direct or safety-grade distance
measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# MLX-VLM creates disabled tqdm progress bars during inference. tqdm's default
# lock is a multiprocessing RLock, which allocates an unnecessary named
# semaphore and can race with Python's resource tracker during shutdown. This
# application coordinates progress only between threads, so use a thread lock.
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None
else:
    _tqdm.set_lock(threading.RLock())


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
DEFAULT_GUIDANCE_RADIUS_FRACTION = 0.075
DEFAULT_QWEN_BACKEND = "auto"
DEFAULT_QWEN_MODEL = None
DEFAULT_QWEN_MLX_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
DEFAULT_QWEN_TRANSFORMERS_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_QWEN_DEVICE = "auto"
DEFAULT_QWEN_DTYPE = "auto"
DEFAULT_QWEN_QUANTIZATION = "auto"
DEFAULT_QWEN_UPLOAD_WIDTH = 864
DEFAULT_QWEN_MAX_NEW_TOKENS = 64
DEFAULT_QWEN_SLOW_SECONDS = 2.0
DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_OCR_FUZZY_THRESHOLD = 0.74
DEFAULT_OCR_MIN_CONFIDENCE = 0.25
DEFAULT_REMOTE_UPLOAD_WIDTH = 960
DEFAULT_REMOTE_JPEG_QUALITY = 85
DEFAULT_REMOTE_TIMEOUT_SECONDS = 15.0
DEFAULT_REMOTE_PREVIEW_WIDTH = 960
DEFAULT_REMOTE_PREVIEW_JPEG_QUALITY = 70
DEFAULT_REMOTE_PREVIEW_INTERVAL = 3
SPEECH_CHANGE_BUFFER_SECONDS = 1.8
HAND_NOT_DETECTED_DELAY_SECONDS = 3.0
HAND_NOT_DETECTED_REPEAT_SECONDS = 5.0
OBJECT_MISSING_CONFIRMATION_ATTEMPTS = 2
MAX_CAMERA_READ_FAILURES = 30
CAMERA_RETRY_DELAY_SECONDS = 0.1
LAPTOP_LOGITECH_CAMERA_INDEX = 0
PI_LOGITECH_CAMERA_INDEX = 0
LOGITECH_FRAME_WIDTH = 1280
LOGITECH_FRAME_HEIGHT = 720
LOGITECH_FRAME_RATE = 30
STATUS_PANEL_HEIGHT = 250


def resolve_qwen_backend(backend: str) -> str:
    """Resolve the portable Qwen backend for the current computer."""
    if backend not in ("auto", "mlx", "transformers"):
        raise ValueError("Qwen backend must be auto, mlx, or transformers")
    apple_silicon = (
        platform.system() == "Darwin"
        and platform.machine().casefold() in ("arm64", "aarch64")
    )
    if backend == "auto":
        return "mlx" if apple_silicon else "transformers"
    if backend == "mlx" and not apple_silicon:
        raise RuntimeError(
            "The MLX Qwen backend requires Apple Silicon. Use the Transformers "
            "backend on Windows or Linux."
        )
    return backend


def default_qwen_model(backend: str) -> str:
    """Return the checkpoint compatible with one resolved Qwen backend."""
    if backend == "mlx":
        return DEFAULT_QWEN_MLX_MODEL
    if backend == "transformers":
        return DEFAULT_QWEN_TRANSFORMERS_MODEL
    raise ValueError(f"Unsupported resolved Qwen backend: {backend}")


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
    confidence: float | None


class GuidanceSpeaker:
    """Repeat actionable guidance asynchronously with one pyttsx3 engine."""

    def __init__(self, debug: bool = False) -> None:
        try:
            import pyttsx3
        except ImportError as error:
            raise RuntimeError(
                "Spoken guidance requires pyttsx3. Install it with "
                "`python -m pip install pyttsx3`."
            ) from error

        self._pyttsx3 = pyttsx3
        self._speech_rate = 50
        self._engine_rate: float | int = self._speech_rate
        self._debug = debug
        self._messages: queue.Queue[
            tuple[int, str, bool] | None
        ] = queue.Queue(maxsize=1)
        self._closed = False
        self._engine = None
        self._cached_sounds: dict[str, object] = {}
        self._engine_error: Exception | None = None
        self._engine_ready = threading.Event()
        self._state_lock = threading.Lock()
        self._next_request_id = 0
        self._outstanding_request_ids: set[int] = set()
        self._current_request_id: int | None = None
        self._current_message: str | None = None
        self._priority_speech_active = threading.Event()
        self._priority_request_id: int | None = None
        self._priority_message: str | None = None
        self._last_debug_suppression: tuple[str, str] | None = None
        self._last_message: str | None = None
        self._last_announcement_at: float | None = None
        self._last_hand_not_detected_at: float | None = None
        self._object_detection_announced = False
        self._object_reannouncement_armed = False
        self._object_announcement_pending = False
        self._object_not_detected_announced = False
        self._object_not_detected_pending = False
        self._hand_was_visible = False
        self._worker = threading.Thread(
            target=self._speech_worker,
            name="guidance-speech",
            daemon=True,
        )
        self._worker.start()

        if not self._engine_ready.wait(timeout=10.0):
            self._closed = True
            raise RuntimeError("Timed out while initializing pyttsx3")
        if self._engine_error is not None:
            self._closed = True
            raise RuntimeError(
                "Could not initialize pyttsx3 with the system speech backend: "
                f"{self._engine_error}"
            ) from self._engine_error

    def _debug_log(self, message: str) -> None:
        if self._debug:
            print(f"[TTS] {message}", flush=True)

    def _prepare_cached_guidance_sound(self, message: str, label: str):
        """Build one in-memory macOS clip for a latency-critical event."""
        if platform.system() != "Darwin":
            return None

        temporary_path: Path | None = None
        try:
            import subprocess
            import tempfile

            from AppKit import NSSound

            with tempfile.NamedTemporaryFile(
                prefix=f"{label}-",
                suffix=".aiff",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            subprocess.run(
                [
                    "say",
                    "-v",
                    "Samantha",
                    "-r",
                    "180",
                    "-o",
                    str(temporary_path),
                    message,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            sound = NSSound.alloc().initWithContentsOfFile_byReference_(
                str(temporary_path),
                False,
            )
            if sound is None:
                raise RuntimeError("NSSound could not load the cached clip")
            sound.setLoops_(False)
            self._debug_log(f"cached {message!r} audio in memory")
            return sound
        except Exception as error:
            self._debug_log(
                f"cached {message!r} audio unavailable: {error}"
            )
            return None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _speech_worker(self) -> None:
        """Own the pyttsx3 engine and speak queued messages on one thread."""
        try:
            # pyttsx3 2.99 still defaults to Apple's deprecated nsss driver,
            # whose run loop can return before NSSpeechSynthesizer finishes.
            # That produces silent later utterances and the misleading error
            # "run loop not started". AVSpeech keeps its loop alive until the
            # native synthesizer reports completion.
            requested_driver = (
                "avspeech" if platform.system() == "Darwin" else None
            )
            try:
                engine = self._pyttsx3.init(requested_driver)
            except Exception:
                if requested_driver is None:
                    raise
                self._debug_log(
                    "avspeech unavailable; falling back to the platform "
                    "pyttsx3 driver"
                )
                engine = self._pyttsx3.init()

            # pyttsx3's AVSpeech adapter uses a normalized multiplier rather
            # than words per minute. 0.7 maps to an AVSpeech rate of 0.35,
            # deliberately close to the user's slower 125-WPM preference.
            self._engine_rate = (
                0.9 if str(engine) == "avspeech" else self._speech_rate
            )
            engine.setProperty("rate", self._engine_rate)

            # Pay AVSpeech's first-utterance setup cost during application
            # startup instead of after the first confirmed object detection.
            # Keeping the warm-up muted preserves the user's audible behavior.
            if str(engine) == "avspeech":
                try:
                    original_volume = engine.getProperty("volume")
                    engine.setProperty("volume", 0.0)
                    engine.say("ready", name="guidance-warmup")
                    engine.runAndWait()
                    engine.setProperty("volume", original_volume)
                    self._debug_log("completed silent AVSpeech warm-up")
                except Exception as error:
                    self._debug_log(f"AVSpeech warm-up skipped: {error}")
                    try:
                        engine.setProperty("volume", 1.0)
                    except Exception:
                        pass
            self._engine = engine
            for cached_message, cached_label in (
                ("Object detected.", "object-detected"),
                ("Detection may take longer.", "slow-detection"),
            ):
                cached_sound = self._prepare_cached_guidance_sound(
                    cached_message,
                    cached_label,
                )
                if cached_sound is not None:
                    self._cached_sounds[cached_message] = cached_sound
            try:
                from importlib.metadata import version

                pyttsx3_version = version("pyttsx3")
            except Exception:
                pyttsx3_version = "unknown"
            self._debug_log(
                "engine ready on speech thread; "
                f"driver={engine}, pyttsx3={pyttsx3_version}, "
                f"requested_rate={self._speech_rate}, "
                f"backend_rate={self._engine_rate}"
            )
        except Exception as error:
            self._engine_error = error
            self._engine_ready.set()
            return

        backend_completion: dict[str, bool] = {}
        connect = getattr(engine, "connect", None)
        if callable(connect):
            connect(
                "started-utterance",
                lambda name: self._debug_log(
                    f"backend started utterance {name!r}"
                ),
            )

            def record_backend_finish(name: str, completed: bool) -> None:
                backend_completion[name] = completed
                self._debug_log(
                    f"backend finished utterance {name!r}; "
                    f"completed={completed}"
                )

            connect("finished-utterance", record_backend_finish)
            connect(
                "error",
                lambda name, exception: self._debug_log(
                    f"backend error for {name!r}: {exception}"
                ),
            )

        self._engine_ready.set()
        try:
            while True:
                request = self._messages.get()
                if request is None:
                    return
                request_id, message, priority = request
                with self._state_lock:
                    self._current_request_id = request_id
                    self._current_message = message
                self._debug_log(
                    f"speaking #{request_id}: {message!r}"
                )
                succeeded = False
                utterance_name = f"guidance-{request_id}"
                try:
                    cached_sound = self._cached_sounds.get(message)
                    if cached_sound is not None:
                        sound = cached_sound
                        sound.stop()
                        sound.setCurrentTime_(0.0)
                        succeeded = bool(sound.play())
                        while succeeded and sound.isPlaying():
                            time.sleep(0.005)
                    else:
                        engine.setProperty("rate", self._engine_rate)
                        engine.say(message, name=utterance_name)
                        engine.runAndWait()
                        succeeded = backend_completion.pop(
                            utterance_name,
                            True,
                        )
                    self._debug_log(
                        f"finished #{request_id}: {message!r}; "
                        f"completed={succeeded}"
                    )
                except Exception as error:
                    print(f"Could not speak guidance: {error}")
                    self._debug_log(
                        f"failed #{request_id}: {message!r}"
                    )
                finally:
                    with self._state_lock:
                        self._current_request_id = None
                        self._current_message = None
                        self._outstanding_request_ids.discard(request_id)
                        if request_id == self._priority_request_id:
                            self._priority_request_id = None
                            self._priority_message = None
                            self._priority_speech_active.clear()
                    if succeeded and message == "Object detected.":
                        # The object event has completed; allow the current
                        # direction to start on the next camera frame instead
                        # of inheriting the normal 1.8-second repeat buffer.
                        self._last_message = None
                        self._last_announcement_at = None
                    if not succeeded and message == "Object detected.":
                        self._object_detection_announced = False
                        self._object_announcement_pending = True
                    if not succeeded and message == "Object not detected.":
                        self._object_not_detected_announced = False
                        self._object_not_detected_pending = True
        finally:
            for cached_sound in self._cached_sounds.values():
                try:
                    cached_sound.stop()
                except Exception:
                    pass
            try:
                engine.stop()
            except Exception:
                pass

    def _clear_pending_messages(self) -> list[tuple[int, str, bool]]:
        removed: list[tuple[int, str, bool]] = []
        while True:
            try:
                request = self._messages.get_nowait()
            except queue.Empty:
                break
            if request is None:
                self._messages.put_nowait(None)
                break
            removed.append(request)
        with self._state_lock:
            for request_id, _, _ in removed:
                self._outstanding_request_ids.discard(request_id)
        return removed

    def _interrupt_current_message(self, expected_message: str | None = None) -> None:
        """Interrupt only an explicitly obsolete current utterance."""
        with self._state_lock:
            current_message = self._current_message
        if expected_message is not None and current_message != expected_message:
            return
        engine = self._engine
        if engine is None or current_message is None:
            return
        try:
            cached_sound = self._cached_sounds.get(current_message)
            if cached_sound is not None:
                cached_sound.stop()
            engine.stop()
        except Exception as error:
            print(f"Could not stop guidance speech: {error}")

    def _start_speech(
        self,
        message: str,
        now: float,
        priority: bool = False,
    ) -> bool:
        """Start one non-blocking message and record its start time."""
        if self._closed:
            return False
        if self._priority_speech_active.is_set() and not priority:
            # A hand warning becomes obsolete as soon as valid reach guidance
            # returns. Allow the new direction to stop and replace it. Other
            # priority notices, such as slow detection, remain protected.
            replacing_stale_hand_warning = (
                self._priority_message == "Hand not detected."
                and (
                    message == "Reach forward."
                    or message.endswith(".")
                    and any(
                        direction in message.casefold()
                        for direction in ("left", "right", "up", "down")
                    )
                )
            )
            if not replacing_stale_hand_warning:
                suppression = (message, self._priority_message or "priority speech")
                if suppression != self._last_debug_suppression:
                    self._debug_log(
                        f"suppressed {message!r} behind "
                        f"{suppression[1]!r}"
                    )
                    self._last_debug_suppression = suppression
                return False
            self.cancel_hand_warning(hand_visible=True)
        elif self._priority_speech_active.is_set() and priority:
            suppression = (message, self._priority_message or "priority speech")
            if suppression != self._last_debug_suppression:
                self._debug_log(
                    f"deferred priority {message!r} behind "
                    f"{suppression[1]!r}"
                )
                self._last_debug_suppression = suppression
            return False

        self._clear_pending_messages()
        with self._state_lock:
            self._next_request_id += 1
            request_id = self._next_request_id
            if priority:
                self._priority_request_id = request_id
                self._priority_message = message
                self._priority_speech_active.set()
            self._outstanding_request_ids.add(request_id)
        if priority:
            self._debug_log(f"queued priority #{request_id}: {message!r}")
        else:
            self._debug_log(f"queued #{request_id}: {message!r}")
        try:
            self._messages.put_nowait((request_id, message, priority))
        except queue.Full as error:
            with self._state_lock:
                if request_id == self._priority_request_id:
                    self._priority_request_id = None
                    self._priority_message = None
                    self._priority_speech_active.clear()
                self._outstanding_request_ids.discard(request_id)
            print(f"Could not speak guidance: {error}")
            return False
        self._last_debug_suppression = None
        self._last_announcement_at = now
        return True

    def wait_until_idle(self, timeout: float = 10.0) -> bool:
        """Wait for queued speech to finish; used by the TTS diagnostic."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._state_lock:
                if not self._outstanding_request_ids:
                    return True
            time.sleep(0.01)
        return False

    def object_detection_announcement_active(self) -> bool:
        """Return whether the confirmed-object message is queued or speaking."""
        with self._state_lock:
            return (
                self._priority_speech_active.is_set()
                and self._priority_message == "Object detected."
            )

    def announce(self, displayed_guidance: str) -> None:
        """Repeat the current instruction at 1.8-second intervals."""
        message = spoken_guidance(displayed_guidance)
        if message is None:
            return

        now = time.monotonic()
        if (
            displayed_guidance == "HAND NOT DETECTED"
            and self._last_hand_not_detected_at is not None
            and now - self._last_hand_not_detected_at
            < HAND_NOT_DETECTED_REPEAT_SECONDS
        ):
            return
        entering_reach = (
            message == "Reach forward." and message != self._last_message
        )
        entering_status = (
            displayed_guidance
            in {
                "HAND NOT DETECTED",
                "STILL SEARCHING",
            }
            and message != self._last_message
        )
        if (
            not entering_reach
            and not entering_status
            and self._last_announcement_at is not None
            and now - self._last_announcement_at < SPEECH_CHANGE_BUFFER_SECONDS
        ):
            return

        # Once the hand has been continuously absent for three seconds, this
        # status must not be lost behind a slow-detection notice or an older
        # directional instruction.
        priority = displayed_guidance == "HAND NOT DETECTED"
        if self._start_speech(message, now, priority=priority):
            self._last_message = message
            if displayed_guidance == "HAND NOT DETECTED":
                self._last_hand_not_detected_at = now

    def announce_once(self, message: str) -> None:
        """Speak one operational notice without changing detector behavior."""
        if self._start_speech(message, time.monotonic(), priority=True):
            self._last_message = message

    def cancel_hand_warning(self, hand_visible: bool) -> None:
        """Stop an obsolete hand warning so directions can resume immediately."""
        if not hand_visible:
            return
        with self._state_lock:
            active_hand_warning = (
                self._priority_speech_active.is_set()
                and self._priority_message == "Hand not detected."
            )
            current_hand_warning = (
                self._current_message == "Hand not detected."
            )
        if active_hand_warning:
            self._clear_pending_messages()
            with self._state_lock:
                self._priority_request_id = None
                self._priority_message = None
                self._priority_speech_active.clear()
            if current_hand_warning:
                self._debug_log("interrupting obsolete hand warning")
                self._interrupt_current_message("Hand not detected.")
        if active_hand_warning or self._last_message == "Hand not detected.":
            self._last_message = None
            self._last_announcement_at = None

    def _preempt_for_object_detection(self) -> None:
        """Make the confirmed-object event the next audible message."""
        self._clear_pending_messages()
        with self._state_lock:
            current_message = self._current_message
            current_request_id = self._current_request_id
            if (
                self._priority_request_id is not None
                and self._priority_request_id != current_request_id
            ):
                self._priority_request_id = None
                self._priority_message = None
                self._priority_speech_active.clear()
            elif (
                current_message is not None
                and current_message != "Object detected."
                and self._priority_request_id == current_request_id
            ):
                # Release the old priority slot before queueing the confirmed
                # detection. The worker will leave the old request registered
                # until engine.stop() makes runAndWait return.
                self._priority_request_id = None
                self._priority_message = None
                self._priority_speech_active.clear()

        if current_message is not None and current_message != "Object detected.":
            self._debug_log(
                f"interrupting {current_message!r} for object detection"
            )
            self._interrupt_current_message(current_message)

        self._last_message = None
        self._last_announcement_at = None

    def announce_object_detection(
        self,
        hand_visible: bool,
        object_detected_now: bool,
    ) -> None:
        """Announce the first detection, then re-arm after the hand disappears."""
        hand_disappeared = self._hand_was_visible and not hand_visible
        self._hand_was_visible = hand_visible
        if hand_disappeared:
            if self._object_detection_announced:
                self._object_reannouncement_armed = True
            self._object_not_detected_announced = False

        should_announce = object_detected_now and (
            not self._object_detection_announced
            or self._object_reannouncement_armed
        )
        if object_detected_now:
            # A confirmed verification starts a fresh visible-object period,
            # so a later failed verification may announce the loss once.
            self._object_not_detected_announced = False
            self._object_not_detected_pending = False
        if should_announce:
            self._object_announcement_pending = True
        if not self._object_announcement_pending:
            return

        self._preempt_for_object_detection()
        if self._start_speech(
            "Object detected.",
            time.monotonic(),
            priority=True,
        ):
            self._object_detection_announced = True
            self._object_reannouncement_armed = False
            self._object_announcement_pending = False

    def announce_object_not_detected(
        self,
        unsuccessful_result_now: bool,
    ) -> None:
        """Announce one unsuccessful result per object-search cycle."""
        if (
            unsuccessful_result_now
            and not self._object_not_detected_announced
        ):
            self._object_not_detected_pending = True
        if not self._object_not_detected_pending:
            return

        if self._start_speech(
            "Object not detected.",
            time.monotonic(),
            priority=True,
        ):
            self._object_not_detected_announced = True
            self._object_not_detected_pending = False

    def close(self) -> None:
        """Stop active speech when the program exits."""
        if self._closed:
            return
        self._closed = True
        self._clear_pending_messages()
        self._interrupt_current_message()
        try:
            self._messages.put_nowait(None)
        except queue.Full:
            self._clear_pending_messages()
            self._messages.put_nowait(None)
        self._worker.join(timeout=5.0)


def run_tts_self_test(debug: bool = True) -> None:
    """Speak every guidance case without loading camera or vision models."""
    speaker = GuidanceSpeaker(debug=debug)

    def speak_and_wait(label: str, action) -> None:
        print(f"TTS self-test: {label}", flush=True)
        action()
        if not speaker.wait_until_idle(timeout=15.0):
            raise RuntimeError(
                f"TTS self-test timed out while waiting for: {label}"
            )

    try:
        speak_and_wait(
            "unsuccessful object detection",
            lambda: speaker.announce_object_not_detected(
                unsuccessful_result_now=True,
            ),
        )
        speak_and_wait(
            "first object detection",
            lambda: speaker.announce_object_detection(
                hand_visible=True,
                object_detected_now=True,
            ),
        )
        speak_and_wait(
            "directional guidance",
            lambda: speaker.announce("MOVE HAND: LEFT | DOWN"),
        )
        speak_and_wait(
            "reach guidance",
            lambda: speaker.announce("ALIGNED: MOVE FORWARD / REACH"),
        )
        speak_and_wait(
            "missing-hand warning",
            lambda: speaker.announce("HAND NOT DETECTED"),
        )
        speaker.cancel_hand_warning(hand_visible=True)
        speak_and_wait(
            "direction after the hand returns",
            lambda: speaker.announce("MOVE HAND: RIGHT"),
        )

        # Reproduce the live re-arm rule: a visible hand disappears, then the
        # next confirmed box must produce a second object announcement.
        speaker.announce_object_detection(
            hand_visible=False,
            object_detected_now=False,
        )
        speak_and_wait(
            "object re-detection after hand disappearance",
            lambda: speaker.announce_object_detection(
                hand_visible=False,
                object_detected_now=True,
            ),
        )
        print("TTS self-test completed successfully.", flush=True)
    finally:
        speaker.close()


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


class QwenSiglipObjectDetector:
    """Ground an object with Qwen and verify its crop with SigLIP and OCR."""

    def __init__(
        self,
        target_name: str,
        model_id: str | None,
        upload_width: int,
        max_new_tokens: int,
        siglip_model_id: str,
        siglip_min_margin: float,
        ocr_verify: bool = False,
        ocr_fuzzy_threshold: float = DEFAULT_OCR_FUZZY_THRESHOLD,
        ocr_min_confidence: float = DEFAULT_OCR_MIN_CONFIDENCE,
        backend: str = DEFAULT_QWEN_BACKEND,
        device: str = DEFAULT_QWEN_DEVICE,
        dtype_name: str = DEFAULT_QWEN_DTYPE,
        quantization: str = DEFAULT_QWEN_QUANTIZATION,
    ) -> None:
        try:
            from qwen_mlx_batch_check import SiglipCropVerifier
            from qwen_mlx_video_check import PaddleTextVerifier
            from qwen_video_check import LocalQwenDetector, pixel_box
        except ImportError as error:
            raise RuntimeError(
                "Qwen + SigLIP guidance requires torch, transformers, and Pillow. "
                "The MLX backend additionally requires mlx-vlm. See the README "
                "setup instructions."
            ) from error

        resolved_backend = resolve_qwen_backend(backend)
        resolved_model_id = model_id or default_qwen_model(resolved_backend)
        self.target_name = target_name
        self.backend = resolved_backend
        self.model_id = resolved_model_id
        self._pixel_box = pixel_box
        self.last_result_reason: str | None = None
        self.ocr_enabled = ocr_verify
        self.last_ocr_match: bool | None = None
        self.last_ocr_text: str | None = None
        print(
            f"Loading local Qwen detector ({resolved_backend}): "
            f"{resolved_model_id}"
        )
        if resolved_backend == "mlx":
            try:
                from qwen_mlx_video_check import LocalMlxQwenDetector
            except ImportError as error:
                raise RuntimeError(
                    "The Apple MLX backend requires mlx-vlm. Install it with "
                    "`python -m pip install --upgrade mlx-vlm`."
                ) from error
            self._qwen = LocalMlxQwenDetector(
                model_id=resolved_model_id,
                upload_width=upload_width,
                max_new_tokens=max_new_tokens,
            )
            runtime_description = "MLX Metal"
        else:
            self._qwen = LocalQwenDetector(
                model_id=resolved_model_id,
                device=device,
                dtype_name=dtype_name,
                upload_width=upload_width,
                max_new_tokens=max_new_tokens,
                quantization=quantization,
            )
            runtime_description = (
                f"PyTorch {self._qwen.device}, {self._qwen.dtype_name}, "
                f"{self._qwen.quantization}"
            )
        print(
            f"Qwen loaded with {runtime_description} in "
            f"{self._qwen.load_seconds:.2f}s."
        )
        print(f"Loading SigLIP verifier: {siglip_model_id}")
        self._siglip = SiglipCropVerifier(
            model_id=siglip_model_id,
            minimum_margin=siglip_min_margin,
        )
        print(
            f"SigLIP loaded and warmed on {self._siglip.device} in "
            f"{self._siglip.load_seconds:.2f}s."
        )
        self._ocr = None
        if ocr_verify:
            self._ocr = PaddleTextVerifier(
                fuzzy_threshold=ocr_fuzzy_threshold,
                min_confidence=ocr_min_confidence,
            )
            print(
                "PaddleOCR verification enabled "
                f"(fuzzy threshold {ocr_fuzzy_threshold:.2f}, "
                f"minimum confidence {ocr_min_confidence:.2f})."
            )

    def detect(self, frame: np.ndarray) -> ObjectDetection | None:
        """Return the first box accepted by SigLIP and optional PaddleOCR."""
        self.last_ocr_match = None
        self.last_ocr_text = None
        self.last_result_reason = None
        detections, _, _, _, _ = self._qwen.detect(
            frame,
            self.target_name,
            target_type="specific" if self.ocr_enabled else "generic",
        )
        if not detections:
            self.last_result_reason = "no_qwen_candidate"
            return None

        candidate = detections[0]
        verification = self._siglip.verify(
            frame,
            candidate,
            self.target_name,
            (self.target_name,),
        )
        if not verification.matched:
            self.last_result_reason = "siglip_rejected"
            print("SigLIP rejected the Qwen box; rescanning a fresh frame.")
            return None

        if self._ocr is not None:
            self.last_ocr_match, self.last_ocr_text = self._ocr.verify(
                frame,
                candidate,
                self.target_name,
            )
            readable_text = self.last_ocr_text or "<no text read>"
            decision = "accepted" if self.last_ocr_match else "rejected"
            print(f"PaddleOCR {decision}: {readable_text}")
            if not self.last_ocr_match:
                self.last_result_reason = "ocr_rejected"
                return None

        frame_height, frame_width = frame.shape[:2]
        left, top, right, bottom = self._pixel_box(
            candidate.box_2d,
            frame_width,
            frame_height,
        )
        self.last_result_reason = "detected"
        return ObjectDetection(
            box=(left, top, right - left, bottom - top),
            # Qwen and SigLIP scores are not calibrated detection confidence.
            confidence=None,
        )


class RemoteObjectDetector:
    """Send Pi camera frames to a computer and receive normalized boxes."""

    def __init__(
        self,
        endpoint: str,
        target_name: str,
        upload_width: int,
        jpeg_quality: int,
        timeout_seconds: float,
    ) -> None:
        parsed_endpoint = urllib.parse.urlparse(endpoint)
        if parsed_endpoint.scheme not in ("http", "https"):
            raise RuntimeError("Remote detector URL must begin with http:// or https://")
        if not parsed_endpoint.netloc:
            raise RuntimeError("Remote detector URL must include a host and port")

        self.endpoint = endpoint
        self.target_name = target_name
        self.upload_width = upload_width
        self.jpeg_quality = jpeg_quality
        self.timeout_seconds = timeout_seconds
        self.last_server_latency_seconds: float | None = None
        self.last_round_trip_seconds: float | None = None
        self.last_result_reason: str | None = None
        self.last_status_message: str | None = None

    def detect(self, frame: np.ndarray) -> ObjectDetection | None:
        """Upload one frame and convert the returned normalized box to pixels."""
        round_trip_started = time.perf_counter()
        self.last_server_latency_seconds = None
        self.last_round_trip_seconds = None
        self.last_result_reason = None
        self.last_status_message = None
        frame_height, frame_width = frame.shape[:2]
        if frame_width > self.upload_width:
            upload_height = max(
                1,
                int(round(frame_height * self.upload_width / frame_width)),
            )
            upload_frame = cv2.resize(
                frame,
                (self.upload_width, upload_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            upload_frame = frame

        encoded_ok, encoded_frame = cv2.imencode(
            ".jpg",
            upload_frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not encoded_ok:
            raise RuntimeError("Could not encode the camera frame as JPEG")

        query = urllib.parse.urlencode({"object": self.target_name})
        separator = "&" if urllib.parse.urlparse(self.endpoint).query else "?"
        request = urllib.request.Request(
            f"{self.endpoint}{separator}{query}",
            data=encoded_frame.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            try:
                details = error.read().decode("utf-8", errors="replace")
            except OSError:
                details = str(error)
            raise RuntimeError(
                f"Computer detector returned HTTP {error.code}: {details}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not reach computer detector at {self.endpoint}: "
                f"{error.reason}"
            ) from error

        self.last_round_trip_seconds = time.perf_counter() - round_trip_started
        reported_latency = payload.get("latency_seconds")
        if isinstance(reported_latency, (int, float)):
            reported_latency = float(reported_latency)
            if math.isfinite(reported_latency) and reported_latency >= 0.0:
                self.last_server_latency_seconds = reported_latency

        reported_reason = payload.get("reason")
        if isinstance(reported_reason, str):
            self.last_result_reason = reported_reason
        reported_message = payload.get("message")
        if isinstance(reported_message, str):
            self.last_status_message = reported_message

        status = payload.get("status")
        if status == "not_detected":
            return None
        if status != "detected":
            raise RuntimeError(
                f"Computer detector returned an invalid status: {status!r}"
            )

        normalized_box = payload.get("bbox_normalized")
        if not isinstance(normalized_box, list) or len(normalized_box) != 4:
            raise RuntimeError(
                "Computer detector response must contain four normalized box values"
            )
        try:
            normalized_left, normalized_top, normalized_right, normalized_bottom = (
                float(value) for value in normalized_box
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "Computer detector returned non-numeric box coordinates"
            ) from error

        left = max(0, min(int(round(normalized_left * frame_width)), frame_width - 1))
        top = max(0, min(int(round(normalized_top * frame_height)), frame_height - 1))
        right = max(left + 1, min(int(round(normalized_right * frame_width)), frame_width))
        bottom = max(
            top + 1,
            min(int(round(normalized_bottom * frame_height)), frame_height),
        )
        return ObjectDetection(
            box=(left, top, right - left, bottom - top),
            confidence=None,
        )


class RemotePreviewSender:
    """Send annotated Pi frames to the computer for visual debugging."""

    def __init__(
        self,
        endpoint: str,
        upload_width: int,
        jpeg_quality: int,
        timeout_seconds: float,
    ) -> None:
        parsed_endpoint = urllib.parse.urlparse(endpoint)
        if parsed_endpoint.scheme not in ("http", "https"):
            raise RuntimeError("Remote preview URL must begin with http:// or https://")
        if not parsed_endpoint.netloc:
            raise RuntimeError("Remote preview URL must include a host and port")
        self.endpoint = endpoint
        self.upload_width = upload_width
        self.jpeg_quality = jpeg_quality
        self.timeout_seconds = timeout_seconds

    def send(self, frame: np.ndarray) -> None:
        """Upload one annotated display frame without waiting for a response body."""
        frame_height, frame_width = frame.shape[:2]
        if frame_width > self.upload_width:
            upload_height = max(
                1,
                int(round(frame_height * self.upload_width / frame_width)),
            )
            upload_frame = cv2.resize(
                frame,
                (self.upload_width, upload_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            upload_frame = frame
        encoded_ok, encoded_frame = cv2.imencode(
            ".jpg",
            upload_frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not encoded_ok:
            raise RuntimeError("Could not encode the debug preview as JPEG")

        request = urllib.request.Request(
            self.endpoint,
            data=encoded_frame.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ):
                pass
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not send debug preview to {self.endpoint}: {error.reason}"
            ) from error


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
    reverse_horizontal: bool = False,
) -> tuple[str, bool]:
    """Return physical movement guidance and alignment state."""
    delta_x = object_center[0] - palm_center[0]
    if reverse_horizontal:
        delta_x = -delta_x
    delta_y = object_center[1] - palm_center[1]
    distance = math.hypot(delta_x, delta_y)
    if distance <= alignment_radius:
        return "ALIGNED: MOVE FORWARD / REACH", True

    directions: list[str] = []
    axis_deadband = max(1.0, alignment_radius * 0.35)
    if delta_x < -axis_deadband:
        directions.append("LEFT")
    elif delta_x > axis_deadband:
        directions.append("RIGHT")
    if abs(delta_y) > alignment_radius:
        directions.append("UP" if delta_y < 0 else "DOWN")

    if not directions:
        directions.append("LEFT" if delta_x < 0 else "RIGHT")
    return "MOVE HAND: " + " | ".join(directions), False


def update_hand_missing_timer(
    hand_visible: bool,
    object_detected: bool,
    missing_started_at: float | None,
    now: float,
) -> tuple[float | None, bool]:
    """Return the continuous missing-hand timer and its three-second state."""
    if hand_visible or not object_detected:
        return None, False
    if missing_started_at is None:
        missing_started_at = now
    return (
        missing_started_at,
        now - missing_started_at >= HAND_NOT_DETECTED_DELAY_SECONDS,
    )


def spoken_guidance(displayed_guidance: str) -> str | None:
    """Return only the directional part of actionable guidance."""
    status_messages = {
        "HAND NOT DETECTED": "Hand not detected.",
        "STILL SEARCHING": "Still searching.",
    }
    if displayed_guidance in status_messages:
        return status_messages[displayed_guidance]
    if displayed_guidance == "ALIGNED: MOVE FORWARD / REACH":
        return "Reach forward."
    if displayed_guidance.startswith("MOVE HAND: "):
        _, separator, directions = displayed_guidance.partition(": ")
        if separator:
            natural_directions = directions.replace(" | ", " and ").lower()
            return f"{natural_directions.capitalize()}."
    return None


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
    label = target_name
    if detection.confidence is not None:
        label += f" {detection.confidence:.0%}"
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
        "--camera-facing",
        choices=("away", "user"),
        default="away",
        help=(
            "Camera direction relative to the user: 'away' for a product-facing "
            "webcam (default), or 'user' for a camera facing the user"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe hand_landmarker.task model",
    )
    parser.add_argument(
        "--object-detector",
        choices=("qwen-siglip", "remote", "yoloe"),
        default="qwen-siglip",
        help=(
            "Object detector: local Qwen 4B plus SigLIP on macOS/Windows "
            "(default), remote computer detection, or YOLOE-26"
        ),
    )
    parser.add_argument(
        "--object-model",
        type=Path,
        default=DEFAULT_OBJECT_MODEL_PATH,
        help="Path to a YOLOE-26 segmentation .pt checkpoint when using yoloe",
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
        "--qwen-backend",
        choices=("auto", "mlx", "transformers"),
        default=DEFAULT_QWEN_BACKEND,
        help=(
            "Qwen runtime: auto selects MLX on Apple Silicon and Transformers "
            "elsewhere (default: auto)"
        ),
    )
    parser.add_argument(
        "--qwen-model",
        default=DEFAULT_QWEN_MODEL,
        help="Override the backend-specific Qwen checkpoint",
    )
    parser.add_argument(
        "--qwen-device",
        choices=("auto", "cuda", "mps", "cpu"),
        default=DEFAULT_QWEN_DEVICE,
        help="PyTorch Qwen device used by the Transformers backend",
    )
    parser.add_argument(
        "--qwen-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default=DEFAULT_QWEN_DTYPE,
        help="PyTorch Qwen dtype used by the Transformers backend",
    )
    parser.add_argument(
        "--qwen-quantization",
        choices=("auto", "4bit", "none"),
        default=DEFAULT_QWEN_QUANTIZATION,
        help="Use bitsandbytes 4-bit Qwen weights on CUDA (default: auto)",
    )
    parser.add_argument(
        "--qwen-upload-width",
        type=int,
        default=DEFAULT_QWEN_UPLOAD_WIDTH,
        help="Image width sent to Qwen (default: 864)",
    )
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=DEFAULT_QWEN_MAX_NEW_TOKENS,
        help="Maximum Qwen response tokens (default: 64)",
    )
    parser.add_argument(
        "--siglip-model",
        default=DEFAULT_SIGLIP_MODEL,
        help="SigLIP checkpoint used to verify Qwen's crop",
    )
    parser.add_argument(
        "--siglip-min-margin",
        type=float,
        default=0.0,
        help="Required SigLIP target margin over competing categories",
    )
    parser.add_argument(
        "--ocr-verify",
        action="store_true",
        help="Verify a branded product using PaddleOCR",
    )
    parser.add_argument(
        "--ocr-fuzzy-threshold",
        type=float,
        default=DEFAULT_OCR_FUZZY_THRESHOLD,
        help="OCR target-word similarity threshold (default: 0.74)",
    )
    parser.add_argument(
        "--ocr-min-confidence",
        type=float,
        default=DEFAULT_OCR_MIN_CONFIDENCE,
        help="Minimum PaddleOCR text confidence (default: 0.25)",
    )
    parser.add_argument(
        "--remote-url",
        default=None,
        help=(
            "Computer detection endpoint when using --object-detector remote, "
            "for example http://10.12.194.2:8000/detect"
        ),
    )
    parser.add_argument(
        "--remote-upload-width",
        type=int,
        default=DEFAULT_REMOTE_UPLOAD_WIDTH,
        help="JPEG width sent to the computer (default: 960)",
    )
    parser.add_argument(
        "--remote-jpeg-quality",
        type=int,
        default=DEFAULT_REMOTE_JPEG_QUALITY,
        help="JPEG quality sent to the computer from 1 to 100 (default: 85)",
    )
    parser.add_argument(
        "--remote-timeout",
        type=float,
        default=DEFAULT_REMOTE_TIMEOUT_SECONDS,
        help="Computer detection request timeout in seconds (default: 15)",
    )
    parser.add_argument(
        "--remote-preview-url",
        default=None,
        help=(
            "Computer debug-preview endpoint; defaults to /preview on the "
            "same host as --remote-url"
        ),
    )
    parser.add_argument(
        "--remote-preview-width",
        type=int,
        default=DEFAULT_REMOTE_PREVIEW_WIDTH,
        help="Width of the annotated laptop preview (default: 960)",
    )
    parser.add_argument(
        "--remote-preview-jpeg-quality",
        type=int,
        default=DEFAULT_REMOTE_PREVIEW_JPEG_QUALITY,
        help="Debug-preview JPEG quality from 1 to 100 (default: 70)",
    )
    parser.add_argument(
        "--remote-preview-every",
        type=int,
        default=DEFAULT_REMOTE_PREVIEW_INTERVAL,
        help="Send one debug-preview frame every N camera frames (default: 3)",
    )
    parser.add_argument(
        "--no-local-display",
        action="store_true",
        help="Do not open an OpenCV window on the Pi; quit with Ctrl-C",
    )
    parser.add_argument(
        "--guidance-radius",
        type=float,
        default=DEFAULT_GUIDANCE_RADIUS_FRACTION,
        help=(
            "Alignment radius as a fraction of the shorter frame side "
            "(default: 0.075)"
        ),
    )
    parser.add_argument(
        "--speak-guidance",
        action="store_true",
        help="Speak guidance changes asynchronously using pyttsx3",
    )
    parser.add_argument(
        "--tts-debug",
        action="store_true",
        help="Print when each TTS message is queued, spoken, or suppressed",
    )
    parser.add_argument(
        "--tts-self-test",
        action="store_true",
        help=(
            "Speak every TTS case without opening the camera or loading "
            "vision models"
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
        help="Display the raw camera image instead of the mirrored preview",
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
    if args.qwen_upload_width <= 0:
        parser.error("--qwen-upload-width must be greater than zero")
    if args.qwen_max_new_tokens <= 0:
        parser.error("--qwen-max-new-tokens must be greater than zero")
    if args.siglip_min_margin < 0.0:
        parser.error("--siglip-min-margin cannot be negative")
    if not 0.0 <= args.ocr_fuzzy_threshold <= 1.0:
        parser.error("--ocr-fuzzy-threshold must be between 0 and 1")
    if not 0.0 <= args.ocr_min_confidence <= 1.0:
        parser.error("--ocr-min-confidence must be between 0 and 1")
    if args.remote_upload_width <= 0:
        parser.error("--remote-upload-width must be greater than zero")
    if not 1 <= args.remote_jpeg_quality <= 100:
        parser.error("--remote-jpeg-quality must be between 1 and 100")
    if args.remote_timeout <= 0.0:
        parser.error("--remote-timeout must be greater than zero")
    if args.remote_preview_width <= 0:
        parser.error("--remote-preview-width must be greater than zero")
    if not 1 <= args.remote_preview_jpeg_quality <= 100:
        parser.error("--remote-preview-jpeg-quality must be between 1 and 100")
    if args.remote_preview_every <= 0:
        parser.error("--remote-preview-every must be greater than zero")
    if args.object_detector == "remote" and not args.remote_url:
        parser.error("--remote-url is required with --object-detector remote")
    if not 0.0 < args.guidance_radius <= 0.5:
        parser.error("--guidance-radius must be greater than 0 and at most 0.5")
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be zero or greater")
    return args


def main() -> None:
    args = parse_args()
    if args.tts_self_test:
        try:
            run_tts_self_test(debug=True)
        except RuntimeError as error:
            raise SystemExit(f"TTS self-test failed: {error}") from error
        return

    hand_model_path = args.model.expanduser().resolve()
    object_model_path = args.object_model.expanduser().resolve()
    if not hand_model_path.is_file():
        raise SystemExit(
            f"MediaPipe model not found: {hand_model_path}\n"
            "Place hand_landmarker.task beside this script or pass --model PATH."
        )
    if args.object_detector == "yoloe" and not object_model_path.is_file():
        raise SystemExit(
            f"Object model not found: {object_model_path}\n"
            "Place yoloe-26n-seg.pt beside this script or pass "
            "--object-model PATH."
        )

    target_name, target_width_meters = resolve_target(
        args.object_name,
        args.object_width_cm,
    )

    if args.object_detector == "qwen-siglip" and args.computer == "pi":
        raise SystemExit(
            "Local Qwen + SigLIP is not supported on this Raspberry Pi. Use "
            "--object-detector remote to call the computer, or use YOLOE."
        )

    try:
        if args.object_detector == "qwen-siglip":
            object_detector = QwenSiglipObjectDetector(
                target_name=target_name,
                model_id=args.qwen_model,
                upload_width=args.qwen_upload_width,
                max_new_tokens=args.qwen_max_new_tokens,
                siglip_model_id=args.siglip_model,
                siglip_min_margin=args.siglip_min_margin,
                ocr_verify=args.ocr_verify,
                ocr_fuzzy_threshold=args.ocr_fuzzy_threshold,
                ocr_min_confidence=args.ocr_min_confidence,
                backend=args.qwen_backend,
                device=args.qwen_device,
                dtype_name=args.qwen_dtype,
                quantization=args.qwen_quantization,
            )
        elif args.object_detector == "remote":
            object_detector = RemoteObjectDetector(
                endpoint=args.remote_url,
                target_name=target_name,
                upload_width=args.remote_upload_width,
                jpeg_quality=args.remote_jpeg_quality,
                timeout_seconds=args.remote_timeout,
            )
            print(f"Computer object detector: {args.remote_url}")
        else:
            object_detector = YoloE26ObjectDetector(
                object_model_path,
                target_name,
                args.object_input_size,
                args.object_confidence,
                args.object_nms_threshold,
            )
    except Exception as error:
        raise SystemExit(
            f"Could not initialize {args.object_detector} object detector:\n{error}"
        ) from error

    remote_preview_sender: RemotePreviewSender | None = None
    if args.object_detector == "remote":
        if args.remote_preview_url:
            remote_preview_url = args.remote_preview_url
        else:
            parsed_remote_url = urllib.parse.urlparse(args.remote_url)
            remote_preview_url = urllib.parse.urlunparse(
                parsed_remote_url._replace(
                    path="/preview",
                    query="",
                    fragment="",
                )
            )
        remote_preview_sender = RemotePreviewSender(
            endpoint=remote_preview_url,
            upload_width=args.remote_preview_width,
            jpeg_quality=args.remote_preview_jpeg_quality,
            timeout_seconds=min(args.remote_timeout, 2.0),
        )
        print(f"Laptop debug preview: {remote_preview_url}")

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
    mirrored_display = args.camera_facing == "user" and not args.no_mirror
    frame_number = 0
    object_detection: ObjectDetection | None = None
    object_visible = False
    object_detector_error: str | None = None
    object_search_enabled = True
    previous_hand_visible = False
    hand_missing_with_object_started_at: float | None = None
    object_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="object-detection",
    )
    object_future: Future[ObjectDetection | None] | None = None
    object_detection_started_at: float | None = None
    slow_warning_announced_for_request = False
    object_has_been_locked = False
    consecutive_object_misses = 0
    rescan_next_frame = False
    last_detection_reason: str | None = None
    last_detection_latency_seconds: float | None = None
    last_server_latency_seconds: float | None = None
    last_round_trip_seconds: float | None = None
    preview_executor: ThreadPoolExecutor | None = None
    preview_future: Future[None] | None = None
    last_preview_error: str | None = None
    if remote_preview_sender is not None:
        preview_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="debug-preview",
        )
    guidance_speaker: GuidanceSpeaker | None = None
    if args.speak_guidance:
        try:
            guidance_speaker = GuidanceSpeaker(debug=args.tts_debug)
        except RuntimeError as error:
            capture.release()
            raise SystemExit(str(error)) from error
        print("Spoken guidance enabled using pyttsx3.")

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

                # End-to-end detection timing begins when the Pi receives the
                # camera frame that will later be submitted for detection.
                frame_received_at = time.perf_counter()
                consecutive_camera_failures = 0
                height, width = camera_frame.shape[:2]
                alignment_radius_pixels = max(
                    1,
                    int(round(args.guidance_radius * min(width, height))),
                )

                # Inference always uses the raw image so XYZ remains in the physical
                # camera frame even when the preview is mirrored for the user.
                rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = time.monotonic_ns() // 1_000_000
                timestamp_ms = max(timestamp_ms, previous_timestamp_ms + 1)
                previous_timestamp_ms = timestamp_ms
                hand_inference_started_at = time.monotonic()
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                selected_index, selected_size = select_foreground_hand(
                    result.hand_landmarks,
                    width,
                    height,
                    args.min_hand_size,
                )
                hand_visible = selected_index is not None
                hand_disappeared = previous_hand_visible and not hand_visible
                previous_hand_visible = hand_visible
                if hand_disappeared and object_detection is not None:
                    object_search_enabled = True
                    object_visible = False
                    print("Hand disappeared; object detection re-armed.")

                # Qwen takes roughly two seconds, so inference runs in a worker
                # while camera and hand tracking remain responsive. Once the
                # object is locked, detection pauses until a visible hand
                # subsequently leaves the camera frame.
                object_detection_ran = False
                object_not_detected_confirmed_now = False
                if object_future is not None and object_future.done():
                    object_detection_ran = True
                    completed_request_started_at = object_detection_started_at
                    try:
                        current_object_detection = object_future.result()
                        object_detector_error = None
                        last_detection_reason = getattr(
                            object_detector,
                            "last_result_reason",
                            None,
                        )
                        object_visible = current_object_detection is not None
                        if current_object_detection is not None:
                            consecutive_object_misses = 0
                            object_detection = current_object_detection
                            object_has_been_locked = True
                            object_search_enabled = False
                            rescan_next_frame = False
                            if guidance_speaker is not None:
                                # Dispatch the success event as soon as Qwen
                                # and its verifier return. Do not wait for box
                                # calculations, drawing, or the rest of this
                                # camera-frame update.
                                guidance_speaker.cancel_hand_warning(
                                    hand_visible=hand_visible,
                                )
                                guidance_speaker.announce_object_detection(
                                    hand_visible=hand_visible,
                                    object_detected_now=True,
                                )
                            left, top, box_width, box_height = (
                                current_object_detection.box
                            )
                            calculated_target_center = (
                                left + box_width // 2,
                                top + box_height // 2,
                            )
                            if completed_request_started_at is not None:
                                # End timing as soon as the Pi has calculated
                                # the target circle. Debug drawing, preview
                                # transfer, and speech are deliberately excluded.
                                last_detection_latency_seconds = (
                                    time.perf_counter()
                                    - completed_request_started_at
                                )
                                timing_breakdown: list[str] = []
                                reported_model_seconds = getattr(
                                    object_detector,
                                    "last_server_latency_seconds",
                                    None,
                                )
                                reported_round_trip_seconds = getattr(
                                    object_detector,
                                    "last_round_trip_seconds",
                                    None,
                                )
                                if reported_model_seconds is not None:
                                    timing_breakdown.append(
                                        f"Mac model {reported_model_seconds:.2f}s"
                                    )
                                if reported_round_trip_seconds is not None:
                                    timing_breakdown.append(
                                        "Pi-Mac round trip "
                                        f"{reported_round_trip_seconds:.2f}s"
                                    )
                                breakdown_suffix = (
                                    f" ({', '.join(timing_breakdown)})"
                                    if timing_breakdown
                                    else ""
                                )
                                print(
                                    "Detection latency "
                                    "(Pi camera frame to calculated target radius): "
                                    f"{last_detection_latency_seconds:.2f}s; "
                                    f"center={calculated_target_center}, "
                                    f"radius={alignment_radius_pixels}px"
                                    f"{breakdown_suffix}"
                                )
                                print(
                                    "Object locked; camera and MediaPipe hand "
                                    "guidance are still running."
                                )
                        elif completed_request_started_at is not None:
                            consecutive_object_misses += 1
                            object_not_detected_confirmed_now = (
                                object_has_been_locked
                                and consecutive_object_misses
                                >= OBJECT_MISSING_CONFIRMATION_ATTEMPTS
                            )
                            if object_not_detected_confirmed_now:
                                # Ignore a single stale or false-negative
                                # inference. Only consecutive misses invalidate
                                # the cached box and pause hand guidance.
                                object_detection = None
                            elif object_has_been_locked:
                                print(
                                    "Object verification miss "
                                    f"{consecutive_object_misses}/"
                                    f"{OBJECT_MISSING_CONFIRMATION_ATTEMPTS}; "
                                    "waiting for confirmation."
                                )
                            no_detection_seconds = (
                                time.perf_counter()
                                - completed_request_started_at
                            )
                            last_detection_latency_seconds = (
                                no_detection_seconds
                            )
                            if last_detection_reason == "siglip_rejected":
                                rescan_next_frame = True
                                print(
                                    "SigLIP rejected the candidate; "
                                    "rescanning the next camera frame."
                                )
                            print(
                                "No-detection response latency "
                                "(Pi camera frame to result): "
                                f"{no_detection_seconds:.2f}s"
                            )
                    except Exception as error:
                        object_visible = False
                        object_detector_error = str(error)
                        last_detection_reason = "error"
                        print(f"Object detection failed: {error}")
                        if completed_request_started_at is not None:
                            failed_seconds = (
                                time.perf_counter()
                                - completed_request_started_at
                            )
                            last_detection_latency_seconds = failed_seconds
                            print(
                                "Failed detection request latency "
                                "(Pi camera frame to error): "
                                f"{failed_seconds:.2f}s"
                            )
                    finally:
                        object_detection_started_at = None
                        last_server_latency_seconds = getattr(
                            object_detector,
                            "last_server_latency_seconds",
                            None,
                        )
                        last_round_trip_seconds = getattr(
                            object_detector,
                            "last_round_trip_seconds",
                            None,
                        )
                    object_future = None

                object_announcement_active = (
                    guidance_speaker is not None
                    and guidance_speaker.object_detection_announcement_active()
                )
                if (
                    object_future is None
                    and object_search_enabled
                    and not object_announcement_active
                    and (
                        rescan_next_frame
                        or hand_disappeared
                        or frame_number % args.object_every == 0
                    )
                ):
                    object_detection_started_at = frame_received_at
                    slow_warning_announced_for_request = False
                    object_future = object_executor.submit(
                        object_detector.detect,
                        camera_frame.copy(),
                    )
                    rescan_next_frame = False
                frame_number += 1

                if mirrored_display:
                    camera_display_frame = cv2.flip(camera_frame, 1)
                else:
                    camera_display_frame = camera_frame.copy()

                # Build the final canvas first, then expose only the lower
                # camera region to every visual-annotation function. This
                # makes it impossible for boxes or landmarks to enter the
                # status panel even though their coordinates start at y=0.
                display_frame = np.full(
                    (height + STATUS_PANEL_HEIGHT, width, 3),
                    (25, 25, 25),
                    dtype=np.uint8,
                )
                camera_display_region = display_frame[
                    STATUS_PANEL_HEIGHT : STATUS_PANEL_HEIGHT + height,
                    :width,
                ]
                camera_display_region[:] = camera_display_frame

                draw_optical_center(camera_display_region)

                palm_xyz: np.ndarray | None = None
                palm_display_center: tuple[int, int] | None = None
                palm_camera_center: tuple[int, int] | None = None
                palm_status = "PALM: not detected"
                if selected_index is not None:
                    image_landmarks = result.hand_landmarks[selected_index]
                    palm_center = palm_center_normalized(image_landmarks)
                    palm_camera_center = _display_normalized_point(
                        palm_center,
                        width,
                        height,
                        False,
                    )
                    palm_display_center = _display_normalized_point(
                        palm_center,
                        width,
                        height,
                        mirrored_display,
                    )
                    draw_hand_and_palm_center(
                        camera_display_region,
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

                detection_elapsed_seconds = (
                    time.perf_counter() - object_detection_started_at
                    if (
                        object_future is not None
                        and object_detection_started_at is not None
                    )
                    else None
                )
                detection_request_slow = (
                    detection_elapsed_seconds is not None
                    and detection_elapsed_seconds >= DEFAULT_QWEN_SLOW_SECONDS
                )
                if (
                    detection_request_slow
                    and not slow_warning_announced_for_request
                ):
                    slow_warning_announced_for_request = True
                    print("Detection may take longer.")
                    if guidance_speaker is not None:
                        guidance_speaker.announce_once(
                            "Detection may take longer."
                        )

                (
                    hand_missing_with_object_started_at,
                    hand_missing_with_object_long_enough,
                ) = update_hand_missing_timer(
                    hand_visible=hand_visible,
                    object_detected=object_detection is not None,
                    missing_started_at=hand_missing_with_object_started_at,
                    now=hand_inference_started_at,
                )
                object_xyz: np.ndarray | None = None
                object_display_center: tuple[int, int] | None = None
                object_camera_center: tuple[int, int] | None = None
                object_status = f"OBJECT ({target_name}): not detected"
                if object_detection is not None:
                    confidence_suffix = (
                        ""
                        if object_detection.confidence is None
                        else f" ({object_detection.confidence:.0%})"
                    )
                    object_camera_center = _display_object_center(
                        object_detection,
                        width,
                        False,
                    )
                    object_display_center = _display_object_center(
                        object_detection,
                        width,
                        mirrored_display,
                    )
                    if target_width_meters is None:
                        object_position_smoother.reset()
                        if object_visible:
                            object_status = (
                                f"OBJECT DETECTED{confidence_suffix}; "
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
                                    f"{object_xyz[2]:+.2f} m"
                                    f"{confidence_suffix}"
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
                    if (
                        last_detection_reason == "siglip_rejected"
                        and object_search_enabled
                    ):
                        object_status = (
                            "OBJECT: STILL SEARCHING; keeping last known box"
                        )
                    draw_object_box(
                        camera_display_region,
                        object_detection,
                        target_name,
                        object_xyz,
                        mirrored_display,
                        cached=not object_visible,
                    )
                else:
                    object_position_smoother.reset()
                    if object_detector_error is not None:
                        object_status = "OBJECT DETECTOR ERROR: see terminal"
                    elif last_detection_reason == "siglip_rejected":
                        object_status = "OBJECT: STILL SEARCHING"
                    elif object_future is not None:
                        object_status = f"OBJECT ({target_name}): searching"

                if detection_request_slow:
                    object_status = "OBJECT: DETECTION MAY TAKE LONGER"

                status_y = 0

                target_label = f"TARGET: {target_name}"
                if target_width_meters is not None:
                    target_label += f" ({target_width_meters * 100.0:.1f} cm wide)"
                _draw_text(
                    display_frame,
                    target_label,
                    status_y + 28,
                    (255, 255, 255),
                    0.66,
                )
                _draw_text(
                    display_frame,
                    palm_status,
                    status_y + 60,
                    (80, 255, 80) if palm_xyz is not None else (0, 165, 255),
                    0.62,
                )
                _draw_text(
                    display_frame,
                    object_status,
                    status_y + 92,
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

                guidance = ""
                aligned = False
                if (
                    palm_camera_center is None
                    and object_camera_center is not None
                    and hand_missing_with_object_long_enough
                ):
                    guidance = "HAND NOT DETECTED"
                elif palm_camera_center is None and object_camera_center is not None:
                    guidance = "GUIDANCE WAITING: show palm"
                elif (
                    object_camera_center is None
                    and last_detection_reason == "siglip_rejected"
                ):
                    guidance = "STILL SEARCHING"
                elif palm_camera_center is None:
                    guidance = "GUIDANCE WAITING: show palm and target object"
                elif object_camera_center is not None:
                    guidance, aligned = guidance_text(
                        palm_camera_center,
                        object_camera_center,
                        alignment_radius_pixels,
                        reverse_horizontal=args.camera_facing == "user",
                    )
                else:
                    guidance = "GUIDANCE WAITING: show target object"

                if guidance_speaker is not None:
                    guidance_speaker.cancel_hand_warning(
                        hand_visible=palm_camera_center is not None,
                    )
                    object_detected_now = (
                        object_detection_ran and object_visible
                    )
                    # Queue the one-time detection event before directional
                    # guidance so a same-frame direction cannot replace it.
                    guidance_speaker.announce_object_detection(
                        hand_visible=palm_camera_center is not None,
                        object_detected_now=object_detected_now,
                    )
                    guidance_speaker.announce_object_not_detected(
                        unsuccessful_result_now=(
                            object_not_detected_confirmed_now
                            and object_detector_error is None
                        ),
                    )
                    guidance_speaker.announce(guidance)

                if object_display_center is not None:
                    # Draw through the camera-only view so a large alignment
                    # circle is clipped at the status-panel boundary.
                    cv2.circle(
                        camera_display_region,
                        object_display_center,
                        alignment_radius_pixels,
                        (80, 255, 80) if aligned else (0, 220, 255),
                        3,
                        cv2.LINE_AA,
                    )
                _draw_text(
                    display_frame,
                    guidance,
                    status_y + 132,
                    (80, 255, 80) if aligned else (0, 220, 255),
                    0.72,
                )

                if target_width_meters is None:
                    _draw_text(
                        display_frame,
                        "OBJECT XYZ/DISTANCE: disabled because width was skipped",
                        status_y + 170,
                        (225, 225, 225),
                        0.58,
                    )
                elif palm_xyz is not None and object_xyz is not None:
                    distance = float(np.linalg.norm(object_xyz - palm_xyz))
                    _draw_text(
                        display_frame,
                        f"PALM-OBJECT DISTANCE: {distance:.2f} m",
                        status_y + 170,
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
                        status_y + 170,
                        (0, 165, 255),
                        0.58,
                    )

                detector_stopwatch_label = (
                    "QWEN + VERIFIER STOPWATCH"
                    if args.object_detector in {"qwen-siglip", "remote"}
                    else "OBJECT DETECTOR STOPWATCH"
                )
                if (
                    object_future is not None
                    and object_detection_started_at is not None
                ):
                    elapsed_detection_seconds = (
                        time.perf_counter() - object_detection_started_at
                    )
                    latency_status = (
                        f"{detector_stopwatch_label}: "
                        f"{elapsed_detection_seconds:.2f} s (running)"
                    )
                elif last_detection_latency_seconds is not None:
                    latency_status = (
                        f"{detector_stopwatch_label}: "
                        f"{last_detection_latency_seconds:.2f} s (last result)"
                    )
                else:
                    latency_status = f"{detector_stopwatch_label}: waiting"
                _draw_text(
                    display_frame,
                    latency_status,
                    status_y + 204,
                    (210, 210, 210),
                    0.54,
                )

                cv2.putText(
                    display_frame,
                    (
                        "Axes: +X left, +Y up, +Z forward - press Q or Esc to quit"
                        if not args.no_local_display
                        else "Axes: +X left, +Y up, +Z forward - Ctrl-C to quit"
                    ),
                    (18, status_y + 236),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(
                    display_frame,
                    (0, STATUS_PANEL_HEIGHT - 1),
                    (width - 1, STATUS_PANEL_HEIGHT - 1),
                    (110, 110, 110),
                    2,
                    cv2.LINE_AA,
                )

                if preview_future is not None and preview_future.done():
                    try:
                        preview_future.result()
                        last_preview_error = None
                    except Exception as error:
                        preview_error = str(error)
                        if preview_error != last_preview_error:
                            print(f"Debug preview unavailable: {preview_error}")
                        last_preview_error = preview_error
                    preview_future = None
                if (
                    remote_preview_sender is not None
                    and preview_executor is not None
                    and preview_future is None
                    and frame_number % args.remote_preview_every == 0
                ):
                    preview_future = preview_executor.submit(
                        remote_preview_sender.send,
                        display_frame.copy(),
                    )

                if not args.no_local_display:
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        key_name = "Q" if key == ord("q") else "Escape"
                        print(
                            f"Hand guidance stopped because {key_name} was "
                            "pressed in the camera window."
                        )
                        break
    except KeyboardInterrupt:
        print("Hand guidance stopped by Ctrl-C.")
    finally:
        if guidance_speaker is not None:
            guidance_speaker.close()
        if preview_executor is not None:
            preview_executor.shutdown(wait=True, cancel_futures=True)
        object_executor.shutdown(wait=True, cancel_futures=True)
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
