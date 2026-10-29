# 180DATeam4

## Programs

- `tetris.py` — Tetris game built with Pygame.
- `hand_object_detection/hand_orientation.py` — Simultaneous palm and
  selected-object tracking with approximate camera-space XYZ. Its models and
  text encoders are kept in the same folder.

MediaPipe tracks the palm, while YOLOE-26n uses the target description entered
in the terminal to detect the object and draw its bounding box. The camera feed
displays the palm and object centers, their estimated XYZ positions, and their
scalar separation when an object width is supplied. It also displays 2-D
left/right/up/down hand guidance.

## Conda setup

Create and activate a Conda environment with Python 3.11:

```bash
conda create --name tetris python=3.11 -y
conda activate tetris
```

On a laptop, install the required packages with:

```bash
python -m pip install opencv-python mediapipe numpy ultralytics
python -m pip install "git+https://github.com/ultralytics/CLIP.git"
```

On a Raspberry Pi 4, use the older ARM64 packages that do not require the AES
CPU feature missing from the Pi 4:

```bash
python -m pip uninstall -y mediapipe opencv-python opencv-contrib-python
python -m pip install --no-cache-dir \
  "numpy==1.26.4" \
  "opencv-contrib-python==4.10.0.84" \
  "mediapipe==0.10.15"
python -m pip install ultralytics
python -m pip install "git+https://github.com/ultralytics/CLIP.git"
```

PyTorch, Ultralytics, and the CLIP text encoder require considerably more disk
space and memory than the previous ONNX-only detector.

## Required local models

Place both models beside `hand_orientation.py`:

- `hand_landmarker.task` — MediaPipe hand-landmark model
- `yoloe-26n-seg.pt` — default YOLOE-26 nano open-vocabulary checkpoint
- `yoloe-26x-seg.pt` — optional extra-large checkpoint

The script requires the selected YOLOE-26 checkpoint to exist locally. With an
internet connection, download it once from inside the detection directory:

```bash
cd hand_object_detection
python -c "from ultralytics import YOLOE; YOLOE('yoloe-26n-seg.pt')"
```

The first text-prompt run also downloads and caches the approximately 254 MB
`mobileclip2_b.ts` text encoder. Finish these downloads before disconnecting
the Pi from the internet. The default detection input size is 320 by 320 pixels.
To try the larger checkpoint without editing the code, run with
`--object-model yoloe-26x-seg.pt`.

Unlike the previous COCO model, YOLOE-26 accepts descriptions that are not
part of a fixed 80-class list. For example:

```bash
python hand_orientation.py --laptop \
  --object "red medicine bottle" \
  --object-width-cm 6
```

## Running the detector

On a Mac:

```bash
cd /Users/jiali/180DATeam4/hand_object_detection
conda activate tetris
python hand_orientation.py --laptop
```

On a Raspberry Pi:

```bash
cd ~/180DATeam4/hand_object_detection
conda activate tetris
python hand_orientation.py --pi
```

Before the camera opens, the program asks only for the target description:

```text
Object to detect: bottle
```

With no width, object detection and the bounding box still work, but object XYZ
and palm-to-object distance are unavailable. Supply a width only when those
metric estimates are needed by using `--object-width-cm`:

```bash
python hand_orientation.py --pi --object bottle --object-width-cm 7
```

Both modes default to Logitech camera index 0. Override it when necessary:

```bash
python hand_orientation.py --laptop --camera 1
python hand_orientation.py --pi --camera 1
```

## Detection and XYZ

For the palm, the red marker is the centroid of the wrist and the four finger
MCP knuckles. MediaPipe world landmarks and OpenCV `solvePnP` estimate the
palm's position relative to the camera.

For the target object, YOLOE-26 converts the description to a text embedding,
produces candidate detections, and selects the strongest matching bounding box.
The script uses its boxes and ignores its optional segmentation masks. The
green box and marker show the detected object. If a real width is supplied, its
depth is estimated from that width and its apparent bounding-box width:

```text
Z = focal_length_pixels * real_object_width / bounding_box_width_pixels
```

Without a supplied width, the program still displays the detected object's box
and center. When a width is supplied and both are detected, it also displays
the object XYZ and straight-line palm-to-object distance.

## On-screen guidance

The program compares the displayed palm center with the center of the detected
object's bounding box. A circle around the object shows the alignment radius.
Outside that circle, the display gives one or two directions such as:

```text
MOVE HAND: LEFT | UP
```

Once the palm center enters the circle, the display changes to:

```text
ALIGNED: MOVE FORWARD / REACH
```

The object is assumed to remain stationary. After at least one successful
detection, the program retains the last object box and center if the hand
occludes it. A yellow box labeled `last known` indicates that cached position,
and guidance continues toward it. Detecting the object again refreshes the
position; restarting the program clears it.

Directions are relative to the displayed preview, including its mirror setting.
The default circle radius is 10% of the shorter frame dimension. Change it with:

```bash
python hand_orientation.py --laptop \
  --object bottle \
  --guidance-radius 0.08
```

Object detection runs every third camera frame by default to reduce Raspberry
Pi load. Change the interval with `--object-every`; using `1` detects on every
frame but may cause substantial lag:

```bash
python hand_orientation.py --pi \
  --object bottle \
  --object-width-cm 7 \
  --object-every 1
```

## Coordinate system

Both estimates use the same camera-centered convention:

- Origin: camera optical center at the lens
- `+X`: camera-left
- `+Y`: camera-up
- `+Z`: forward from the camera into the scene
- Units: meters

Inference always uses the raw camera image, so XYZ signs do not change when the
preview is mirrored. Use `--no-mirror` for a preview that directly matches the
physical camera-coordinate directions.

The Logitech C270's nominal 55-degree diagonal field of view is used by
default. Override it for another camera:

```bash
python hand_orientation.py --laptop --diagonal-fov 55
```

When a width is supplied, print palm XYZ, object XYZ, and scalar separation as
CSV-style terminal output:

```bash
python hand_orientation.py --pi \
  --object bottle \
  --object-width-cm 7 \
  --print-xyz
```

The program detects up to four hands and selects the hand with the largest
visible palm span. Hands smaller than 10% of the frame are ignored so that a
distant person's hand is less likely to be selected:

```bash
python hand_orientation.py --pi \
  --object bottle \
  --object-width-cm 7 \
  --min-hand-size 0.15
```

## Accuracy and safety limits

The C270 is a monocular RGB camera and does not measure depth. Palm depth is
inferred from MediaPipe's estimated hand scale. Object depth is available only
when a width is supplied and is inferred from that width. Rotation, partial
occlusion, inaccurate product dimensions, lens distortion, and camera
calibration errors can all produce incorrect XYZ values.

This prototype must not be treated as safety-grade positioning or used by
itself for navigation or obstacle avoidance. A calibrated stereo/depth camera
and independent safety checks are recommended before it is used as an
assistive grabbing system. Entering the 2-D alignment circle does not verify
object depth, a clear path, or that the object is safely within reach. Moving
the object or camera while a cached position is in use makes the guidance stale.

Press `q` or Escape to quit. When finished, leave Conda with:

```bash
conda deactivate
```
