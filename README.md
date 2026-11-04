# 180DATeam4

## Programs

- `tetris.py` — Tetris game built with Pygame.
- `hand_orientation.py` — Live MediaPipe hand-orientation display.
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

## Hand-orientation setup with Conda

Create a Conda environment with Python 3.11:

```bash
conda create --name tetris python=3.11 -y
conda activate tetris
```

Install the required packages inside the activated environment:

```bash
python -m pip install opencv-python mediapipe numpy
```

Run the orientation display from the project directory:

```bash
cd /Users/jiali/180DATeam4
python hand_orientation.py
```

The included hand model is loaded locally; the script does not download any
files. Hold one straight hand with all four fingertips aimed at the camera.
Pointer/index finger up and pinky down is `0 degrees`; clockwise rotation in
the displayed image is positive and counterclockwise rotation is negative.
Press `q` or Escape to quit.

You can adjust the pointing tolerance or select a different camera:

```bash
python hand_orientation.py --max-pointing-angle 25 --camera 1
```

When finished, leave the Conda environment with:

```bash
conda deactivate
```
