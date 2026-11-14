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
python -m pip install opencv-python mediapipe numpy ultralytics
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
