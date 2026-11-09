# CVS Store Mapping Guide — Field Manual & Protocol

This guide provides step-by-step instructions for surveying, mapping, and digitizing a retail store (such as CVS Pharmacy) for the blind assistive shopping assistant.

---

## 1. Overview & Conceptual Model

The navigation system uses a **Topological Waypoint Graph with Geometric Distance & Heading**:
- **Nodes (Waypoints):** Physical locations in the store (e.g., Entrance, Aisle Intersections, Specific Product Shelves, Checkout).
- **Edges (Pathways):** Navigable corridors connecting two nodes, defined by physical distance (in meters or step count) and compass heading ($0^\circ$ to $360^\circ$).
- **Visual Landmarks:** Text and visual cues (like hanging overhead aisle signs: *"AISLE 4 SKIN CARE"*) that the wearable camera scans via OCR to auto-correct drift.
- **Product Placements:** Exact shelf coordinates (left/right side, top/eye-level/middle/bottom) and descriptive metadata for items.

```
       [Aisle 4 Entrance]  <--- (5.5m @ 270°) ---  [Main Hallway]
              |                                          ^
        (4.5m @ 0°)                                      | (4.5m @ 0°)
              v                                          |
    [Aisle 4 - Moisturizers]                       [Front Lobby]
     (CeraVe, Neutrogena)                                ^
                                                         | (3.5m @ 0°)
                                                   [Store Entrance] (0, 0)
```

---

## 2. Pre-Trip Preparation

### Tools to Bring:
1. **Smartphone:** 
   - Camera for photographing aisle signs, shelves, and products.
   - iOS *Measure* app or Android *AR Ruler* (or simple stride-based pacing).
   - Compass app (to note store orientation relative to magnetic North).
2. **Notepad / Digital Notes:** To jot down pace counts, shelf sides, and verbatim sign text.
3. **Pacing Calibration:**
   - Measure your natural stride before heading out.
   - Walk 10 normal steps and measure the total distance in meters.
   - Formula: $\text{Step Length} = \frac{\text{Total Distance in Meters}}{10}$. (Default is typically **0.70 m**).

---

## 3. On-Site Field Protocol (Walking the Store)

### Step 1: Establish Origin $(0, 0)$ & Heading Reference
- Stand at the **main entrance doors** facing straight inside the store.
- Assign this point as `entrance` at coordinate $(X=0.0, Y=0.0)$.
- The direction directly into the store is assigned as **North ($0^\circ$)**:
  - $0^\circ$ = Straight forward into the store ($+Y$).
  - $90^\circ$ = Turning right ($+X$).
  - $180^\circ$ = Turning around towards the entrance ($-Y$).
  - $270^\circ$ = Turning left ($-X$).

### Step 2: Map the Main Walkways & Intersections
Walk forward along the main central concourse. For each intersection where an aisle branches off:
1. Note the number of paces/steps from the entrance (e.g., 5 steps $\approx 3.5$m).
2. Create a node ID: e.g. `front_lobby`, `main_hallway_center`.
3. Note any major junction landmarks (e.g., shopping basket stand, pharmacy pickup sign).

### Step 3: Map the Target Aisle (e.g. Aisle 4 - Skincare)
1. At the aisle entrance, look up and record the **exact text on the hanging overhead sign**:
   - *Example:* `AISLE 4 SKIN CARE SUN CARE FACIAL CLEANSERS`.
   - *Why this matters:* The wearable camera continuously looks for this text to reset IMU drift and announce: *"Position confirmed: Entered Aisle 4"*.
2. Turn into the aisle (note the heading, e.g. $0^\circ$).
3. Walk to the target product section (e.g., facial moisturizers) and count paces (e.g., 6 paces $\approx 4.5$m).
4. Create a node: `aisle_4_skincare`.

### Step 4: Catalog the Target Products at the Shelf
For each item in your test scenario, record:
- **Product Name:** (e.g., *"CeraVe Moisturizing Cream"*)
- **Brand:** (e.g., *"CeraVe"*)
- **Category:** (e.g., *"moisturizer"*)
- **Shelf Side:** `left` or `right` (relative to facing forward down the aisle).
- **Shelf Height:**
  - `top` (above eye level, reaching up)
  - `eye_level` (directly in forward gaze)
  - `middle` (waist/chest level)
  - `bottom` (shin/knee level)
- **Visual Description:** Container shape, primary colors, packaging details (used later by Gemini Flash and visual verification):
  - *Example:* *"16 oz blue and white circular tub with pump"*.

### Step 5: Map Checkout & Exit Routes
- Map the path from the back/aisles to the checkout registers (`checkout_registers`).
- Record overhead sign text: `CHECKOUT / SELF-CHECKOUT`.

---

## 4. Converting Field Notes into the Map JSON

You can use the built-in interactive CLI tool or write the JSON directly.

### Option A: Using the Interactive CLI Wizard
Run the CLI tool in your terminal:
```bash
python tools/cvs_map_tool.py create --output data/my_cvs_map.json
```
The wizard will prompt you:
1. Store name and address.
2. Store entrance waypoint.
3. Additional aisle waypoints, overhead sign text, and products.
4. Distances (in paces or meters) and compass headings between waypoints.

---

### Option B: Editing JSON Directly
Store maps are saved in `data/<store_id>.json`. Here is the schema template:

```json
{
  "metadata": {
    "store_name": "CVS Pharmacy - Westwood Blvd",
    "store_id": "cvs_westwood_0482",
    "address": "1001 Westwood Blvd, Los Angeles, CA 90024",
    "default_step_length_m": 0.70,
    "north_reference_deg": 0.0,
    "version": "1.0.0"
  },
  "nodes": {
    "entrance": {
      "node_id": "entrance",
      "name": "Store Entrance",
      "node_type": "entrance",
      "x": 0.0,
      "y": 0.0,
      "landmarks": [
        {
          "landmark_id": "lm_entrance",
          "landmark_type": "door",
          "text": "WELCOME TO CVS PHARMACY"
        }
      ],
      "products": []
    },
    "aisle_4_entrance": {
      "node_id": "aisle_4_entrance",
      "name": "Aisle 4 Entrance",
      "node_type": "intersection",
      "x": -5.5,
      "y": 8.0,
      "landmarks": [
        {
          "landmark_id": "lm_aisle_4_sign",
          "landmark_type": "overhead_aisle_sign",
          "text": "AISLE 4 SKIN CARE SUN CARE"
        }
      ],
      "products": []
    },
    "aisle_4_skincare": {
      "node_id": "aisle_4_skincare",
      "name": "Aisle 4 - Moisturizers",
      "node_type": "shelf_target",
      "x": -5.5,
      "y": 12.5,
      "landmarks": [],
      "products": [
        {
          "product_id": "prod_cerave",
          "name": "CeraVe Moisturizing Cream",
          "category": "moisturizer",
          "keywords": ["cerave", "lotion", "cream", "dry skin"],
          "shelf_side": "right",
          "shelf_height": "middle",
          "brand": "CeraVe",
          "description": "16 oz blue and white tub"
        }
      ]
    }
  },
  "edges": [
    {
      "from_node": "entrance",
      "to_node": "aisle_4_entrance",
      "distance_meters": 9.7,
      "heading_deg": 325.0,
      "bidirectional": true,
      "notes": "Walk northwest through front concourse"
    },
    {
      "from_node": "aisle_4_entrance",
      "to_node": "aisle_4_skincare",
      "distance_meters": 4.5,
      "heading_deg": 0.0,
      "bidirectional": true,
      "notes": "Proceed straight down Aisle 4"
    }
  ]
}
```

---

## 5. Map Validation & Simulation

Once your map is saved, use `tools/cvs_map_tool.py` to test and verify it before live user testing:

### 1. Check for Structural & Topological Errors
```bash
python tools/cvs_map_tool.py validate --map data/sample_cvs_map.json
```
- Checks that all nodes are connected.
- Ensures no orphaned nodes exist that cannot be reached from the entrance.
- Confirms valid heading angles and step calculations.

### 2. Test Product Keyword & Natural Language Search
```bash
python tools/cvs_map_tool.py search --query "Where is the moisturizer?" --map data/sample_cvs_map.json
python tools/cvs_map_tool.py search --query "sunscreen" --map data/sample_cvs_map.json
python tools/cvs_map_tool.py search --query "ibuprofen" --map data/sample_cvs_map.json
```

### 3. Simulate Turn-by-Turn Navigation & Audio Prompts
```bash
python tools/cvs_map_tool.py navigate --from entrance --query "moisturizer" --map data/sample_cvs_map.json
```
**Example Generated Audio:**
> 🔊 *"Route planned to Aisle 4 - Facial Moisturizers & Lotions for CeraVe Moisturizing Cream. Total distance is approximately 25 steps, about 18.0 meters."*  
> 🔊 *"Continue straight for 5 steps towards Front Concourse & Shopping Baskets."*  
> 🔊 *"Turn left, then walk forward 8 steps towards Aisle 4 Entrance. You should see overhead sign: 'AISLE 4 SKIN CARE SUN CARE FACIAL CLEANSERS'."*  
> 🔊 *"Turn right, then walk forward 6 steps towards Aisle 4 - Facial Moisturizers & Lotions."*  
> 🔊 *"You have arrived at Aisle 4 - Facial Moisturizers & Lotions. CeraVe Moisturizing Cream is on your right, at middle height. Turn 90 degrees right to face the shelf directly. When ready, ask 'What is on the shelf?' to scan."*

### 4. Export Visual 2D Diagram & SVG
```bash
python tools/cvs_map_tool.py visualize --map data/sample_cvs_map.json --svg data/store_map.svg
```
Generates both an ASCII directory and a scalable color-coded 2D SVG map (`data/store_map.svg`) showing all aisles, nodes, and product shelves.

---

## 6. Practical Tips & Troubleshooting in the Field

1. **Metal Shelving & Compass Distortion:**  
   Retail stores have heavy steel shelving and refrigeration units that distort raw magnetometer readings.  
   *Solution:* The system uses **relative gyroscope integration** for turn detection ($90^\circ$ turns) rather than pure absolute magnetic compass readings, and uses the **camera reading overhead aisle signs** as ground-truth drift resets.
2. **Floor Display Stands (Cardboard Cutouts):**  
   Stores frequently place temporary cardboard promotional stands in the center of main concourses.  
   *Solution:* Always map pathways along the open pedestrian walking lane (allow ~1.5m clearance). Add notes in `notes` attribute of the edge.
3. **Pacing Differences Between Team Members:**  
   Different people have different step lengths. Adjust `default_step_length_m` in the map metadata (or pass custom step length) if the person wearing the rig has longer or shorter strides.

