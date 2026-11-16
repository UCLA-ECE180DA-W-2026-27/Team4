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

Install the required packages:

```bash
python -m pip install opencv-python mediapipe numpy ultralytics google-genai
```

Download the YOLOE model once:

```bash
cd hand_object_detection
python -c "from ultralytics import YOLOE; YOLOE('yoloe-26n-seg.pt')"
```

Run on a laptop with an outward-facing camera:

```bash
python hand_orientation.py \
  --laptop \
  --camera-facing away \
  --object "pasta box" \
  --speak-guidance
```

Use `--camera 1`, `--camera 2`, and so on to select another camera. Object
names containing spaces must be enclosed in quotes.

Run on a Raspberry Pi after installing compatible dependencies:

```bash
cd ~/180DATeam4/hand_object_detection
python hand_orientation.py \
  --pi \
  --camera-facing away \
  --object "pasta box" \
  --speak-guidance
```

Press `q` or Escape to close the camera display. When finished:

```bash
conda deactivate
```

## Local Qwen video object-detection benchmark

`hand_object_detection/qwen_video_check.py` runs the official
`Qwen/Qwen3-VL-2B-Instruct` model locally. Frames and prompts remain on the Mac;
only the initial model download requires internet access. The model is loaded
once and reused across samples, and loading time is reported separately from
per-frame inference.

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
local Hugging Face cache. On Apple Silicon, the script uses Metal automatically
and retains the model between frames. It writes annotated frames, a review MP4,
and `results.json` under `hand_object_detection/qwen_video_test_output/`.

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
with offline text verification, add `--ocr-verify`. The verifier uses macOS
Vision OCR on the corresponding crop
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
boxes are recorded in `results.json`. This OCR mode is macOS-only.
