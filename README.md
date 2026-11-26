# 180DATeam4

## Programs

- `tetris.py` — Tetris game built with Pygame.
- `hand_object_detection/hand_orientation.py` — Detects the user's palm and a selected object, estimates their 3D positions relative to the camera, and provides visual or spoken hand guidance.
- `navigation/` — Indoor store navigation engine, topological map schema, Dijkstra pathfinding, and IMU/visual dead reckoning.
- `vision/` — Gemini Flash Multimodal VLM shelf inspection, conversational product Q&A, and visual servoing target lock.
- `tools/cvs_map_tool.py` — Interactive CLI to create, validate, simulate navigation, and visualize CVS store maps.
- `tools/shelf_qa_tool.py` — Interactive shelf exploration & Q&A assistant (camera snapshot, VLM analysis, Q&A dialogue, target locking).
- `docs/STORE_MAPPING_GUIDE.md` — Complete field guide on how to survey and create a CVS store map.

## CVS Store Mapping & Navigation

### 1. Validate Store Map
```bash
python tools/cvs_map_tool.py validate --map data/sample_cvs_map.json
```

### 2. Search for Products (Voice Query Simulation)
```bash
python tools/cvs_map_tool.py search --query "Where is the moisturizer?" --map data/sample_cvs_map.json
```

### 3. Simulate Turn-by-Turn Spoken Navigation for Blind User
```bash
python tools/cvs_map_tool.py navigate --from entrance --query "moisturizer" --map data/sample_cvs_map.json
```

### 4. Visualize Store Map (ASCII and Scalable 2D SVG)
```bash
python tools/cvs_map_tool.py visualize --map data/sample_cvs_map.json --svg data/store_map.svg
```

### 5. Create a New Store Map Interactively
```bash
python tools/cvs_map_tool.py create --output data/my_cvs_map.json
```
See [`docs/STORE_MAPPING_GUIDE.md`](docs/STORE_MAPPING_GUIDE.md) for full surveying instructions.

## Shelf Vision & Conversational Q&A (Gemini Flash)

### 1. Run Interactive Shelf Q&A (Offline / Sample Shelf Mode)
```bash
python tools/shelf_qa_tool.py --image data/sample_cvs_shelf.jpg
```

### 2. Run with Live Google Gemini Flash API Key
```bash
python tools/shelf_qa_tool.py --image data/sample_cvs_shelf.jpg --api-key YOUR_GEMINI_API_KEY
```
Or export your key once in your terminal:
```bash
export GEMINI_API_KEY="your-api-key-here"
python tools/shelf_qa_tool.py --image data/sample_cvs_shelf.jpg
```

### 3. Capture Live Frame from Wearable / Attached USB Camera
```bash
python tools/shelf_qa_tool.py --camera 0
```
This automatically captures a snapshot, announces the spoken shelf summary, allows conversational questions (*"Which is oil-free?"*, *"Tell me about CeraVe"*), locks onto your chosen target, and outputs the exact pixel bounding box and crosshairs in `data/shelf_detected.jpg`.

## Hand and object guidance with Conda

Create and activate a Conda environment with Python 3.11:

```bash
conda create --name tetris python=3.11 -y
conda activate tetris
```

Install the common packages:

```bash
python -m pip install opencv-python mediapipe numpy ultralytics
python -m pip install paddlepaddle "paddleocr[all]"
```

On an Apple Silicon Mac, install the MLX Qwen and SigLIP dependencies:

```bash
python -m pip install --upgrade mlx-vlm torch torchvision transformers pillow
```

Run the bridged Qwen 4B + SigLIP detector and hand guidance. Qwen automatically
uses the Apple MLX backend. The first run downloads both models; later runs use
the local cache:

```bash
cd hand_object_detection
python hand_orientation.py \
  --laptop \
  --camera-facing away \
  --object "soup can" \
  --speak-guidance
```

Qwen searches until it confirms the first object box, and SigLIP verifies the
category. Detection runs in the background so the camera and MediaPipe hand
tracking remain responsive. The confirmed box is retained while the user
reaches; object detection is re-armed only after the tracked hand disappears.

Use `--camera 1`, `--camera 2`, and so on to select another camera. Object
names containing spaces must be enclosed in quotes.

### Windows Qwen setup

The Windows path uses the official `Qwen/Qwen3-VL-4B-Instruct` checkpoint with
PyTorch/Transformers. An NVIDIA GPU is strongly recommended. Install the
CUDA-enabled PyTorch build selected for the computer at
<https://pytorch.org/get-started/locally/>, then install the remaining packages
from PowerShell:

```powershell
cd hand_object_detection
python -m pip install -r requirements-windows.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Run object detection locally with the Windows camera:

```powershell
python hand_orientation.py --laptop --camera-facing away --object "soup can" --speak-guidance
```

Or run only the computer detector for the split Pi/computer architecture:

```powershell
python object_detection_server.py --host WINDOWS_USB_IP --object "cereal"
```

On Windows, `--qwen-backend auto` selects Transformers, `--qwen-device auto`
selects CUDA when available, and `--qwen-quantization auto` loads Qwen in 4-bit
mode through bitsandbytes. A CPU fallback is available with
`--qwen-device cpu --qwen-quantization none`, but it is unlikely to meet the
interactive latency target.

Local Qwen is intentionally run on the Mac or Windows computer, not the
Raspberry Pi. To run object detection directly on the Pi after installing
compatible MediaPipe and Ultralytics dependencies, select the retained YOLOE
fallback:

```bash
cd ~/180DATeam4/hand_object_detection
python hand_orientation.py \
  --pi \
  --object-detector yoloe \
  --camera-facing away \
  --object "pasta box" \
  --speak-guidance
```

### Split Pi hand guidance and computer object detection

For the intended two-device architecture, connect the Logitech camera to the
Pi. The Pi keeps each frame for MediaPipe hand tracking and sends a 960-pixel
JPEG copy over the private USB-network link. The computer runs Qwen + SigLIP
and optionally verifies branded-product text with PaddleOCR before
returning a normalized bounding box. The Pi caches that box for guidance and
requests another detection only after the hand disappears.

Start the detector on the computer, using the computer's USB-interface IP:

```bash
cd /Users/jiali/180DATeam4/hand_object_detection
conda activate tetris

python object_detection_server.py \
  --host 10.12.194.2 \
  --object "cereal"
```

For an exact branded product, require the PaddleOCR verifier:

```bash
python object_detection_server.py \
  --host 10.12.194.2 \
  --object "Campbell's soup" \
  --ocr-verify
```

OCR is intentionally optional: use it for targets with distinctive visible
brand or product text, but leave it off for generic targets such as `cereal`
or `reusable bottle`. PaddleOCR uses the lightweight PP-OCRv5 Mobile detection
model and English recognition model. The first OCR-enabled run downloads the
models; later runs reuse the local cache.

Then start hand guidance on the Pi with the same object name:

```bash
cd ~/180DATeam4/hand_object_detection
conda activate tetris

python hand_orientation.py \
  --pi \
  --camera 0 \
  --camera-facing away \
  --object-detector remote \
  --remote-url http://10.12.194.2:8000/detect \
  --object "cereal" \
  --speak-guidance \
  --no-local-display
```

Replace `10.12.194.2` if the computer has a different address on the USB
interface. The computer opens a live debug window showing the Pi's annotated
camera feed, hand landmarks, returned object box, current guidance, and
end-to-end detection latency. Timing begins when the Pi reads the camera frame
selected for detection and ends when the returned box has been converted into
the target center and radius. Debug drawing, preview upload, and speech are not
included. The terminal also reports Mac model and Pi-to-Mac round-trip timing
as a breakdown. Press `q` in that window to stop the computer server; stop the
Pi process with Ctrl-C. Use `--no-preview` on the computer only when no debug
window is needed.

To start both processes from one Mac command instead, run:

```bash
cd /Users/jiali/180DATeam4/hand_object_detection
bash start_pi_hand_guidance.sh "cereal"
```

Enable PaddleOCR from the combined launcher for a branded product:

```bash
bash start_pi_hand_guidance.sh "Campbell's soup" --ocr
```

The launcher starts the computer server, waits for the models to become ready,
then starts the Pi process over SSH. It uses computer USB IP `10.12.194.2`, Pi
address `piuser@10.12.194.1`, and the existing Pi Conda environment by default.
Press Ctrl-C in the terminal to stop the combined run.

Press `q` or Escape to close the camera display. When finished:

```bash
conda deactivate
```

## Local Qwen video object-detection benchmark

`hand_object_detection/qwen_video_check.py` runs the official
`Qwen/Qwen3-VL-2B-Instruct` model locally through PyTorch/Transformers. Frames
and prompts remain on the computer; only the initial model download requires
internet access. The model is loaded once and reused across samples, and
loading time is reported separately from per-frame inference.

Install the local inference dependencies in the Conda environment:

```bash
python -m pip install "transformers>=4.57" torch torchvision pillow
```

Run the same four-frame Campbell's soup comparison used for Gemini:

```bash
cd hand_object_detection
python qwen_video_check.py \
  --object "Campbell's soup" \
  --start-seconds 24 \
  --end-seconds 30 \
  --sample-every-seconds 2 \
  --no-open
```

The first run downloads several gigabytes of weights. Later runs reuse the
local Hugging Face cache. The script automatically selects CUDA on Windows,
Metal on Apple Silicon, or CPU as a slower fallback. Windows CUDA uses 4-bit
bitsandbytes by default. It writes annotated frames, a review MP4, and
`results.json` under `hand_object_detection/qwen_video_test_output/`.

### Four-bit MLX benchmark on Apple Silicon

`qwen_mlx_video_check.py` runs the same benchmark with the 4-bit
`mlx-community/Qwen3-VL-2B-Instruct-4bit` checkpoint and the Apple-optimized
MLX backend. It is kept separate from the PyTorch script for direct comparison.

Install MLX-VLM inside the activated Conda environment:

```bash
python -m pip install --upgrade mlx-vlm
```

Run the four-frame comparison at 1280-pixel input width:

```bash
cd hand_object_detection
python qwen_mlx_video_check.py \
  --object "Campbell's soup" \
  --start-seconds 24 \
  --end-seconds 30 \
  --sample-every-seconds 2 \
  --upload-width 1280 \
  --no-open
```

The first run downloads the approximately 1.8 GB quantized checkpoint. The
model remains loaded across sampled frames and later runs reuse the cache.

Test several generic objects without OCR in one run:

```bash
python qwen_mlx_batch_check.py
```

By default, this checks `water`, `cereal`, `soup can`, and `ham` on the frame
at 26 seconds using a 960-pixel input and the 4-bit
`mlx-community/Qwen3-VL-4B-Instruct-4bit` checkpoint. Generic mode accepts any
brand and identifies the requested category from appearance, packaging, and
context without requiring matching brand text. The model is loaded only once.
The first run downloads the 4B checkpoint. The output folder contains a
labeled image for every object, a combined `batch_summary.jpg`, and a
`results.json` file with each inference time. Change the objects or timestamp
without changing the script:

```bash
python qwen_mlx_batch_check.py \
  --objects water cereal "soup can" ham \
  --time-seconds 26
```

Add local SigLIP 2 crop verification to reject Qwen boxes that resemble a
different grocery category:

```bash
python qwen_mlx_batch_check.py \
  --objects water cereal "soup can" ham \
  --time-seconds 26 \
  --upload-width 960 \
  --siglip-verify \
  --output-dir qwen_video_test_output/generic_4b_960_siglip
```

The user still enters short names such as `cereal`. The script automatically
expands them into visual descriptions and compares each Qwen crop with common
competing grocery categories such as tuna, soup, water, and ham. The first run
downloads `google/siglip2-base-patch16-224`; later runs reuse the local cache.
Qwen time, SigLIP time, category scores, and the final accept/reject decision
are saved in `results.json`.

Test several branded products on the same frame with one Qwen and PaddleOCR
model load:

```bash
python qwen_mlx_batch_check.py \
  --video ../test1.mp4 \
  --objects "Campbell's soup" "StarKist tuna" "Raisin Bran" \
  --time-seconds 26 \
  --upload-width 960 \
  --ocr-verify \
  --output-dir qwen_video_test_output/test1_branded_paddleocr
```

This treats every target as a specific product, saves an annotated image for
each brand, and records the recognized text, OCR match, and OCR latency in
`results.json`.

Measure accuracy across the 12 labeled stationary-frame cases in `test1.mp4`:

```bash
python qwen_siglip_accuracy_check.py
```

The benchmark loads Qwen and SigLIP once, tests all four categories at 0, 20,
and 26 seconds, then reports the confusion matrix, accuracy, precision, recall,
specificity, F1 score, and correct-detection latency. It saves every annotated
case, `accuracy_summary.jpg`, and a detailed `results.json`. Use `--cases-json`
to evaluate another labeled video without editing the script.

OCR verification is disabled by default, which is appropriate for generic
objects that may not contain useful text. For faster 960-pixel Qwen proposals
with offline text verification, add `--ocr-verify`. The verifier uses PaddleOCR
PP-OCRv5 Mobile on the corresponding crop
from the original full-resolution frame, then accepts the box only when every
distinctive target word matches. Rejected candidates are drawn in orange in
the review video.

```bash
python qwen_mlx_video_check.py \
  --object "Campbell's soup" \
  --start-seconds 24 \
  --end-seconds 26 \
  --sample-every-seconds 2 \
  --upload-width 960 \
  --ocr-verify \
  --output-dir qwen_video_test_output/test1_campbells_mlx_960_ocr \
  --no-open
```

OCR text, match decisions, OCR latency, Qwen candidate boxes, and accepted
boxes are recorded in `results.json`.
