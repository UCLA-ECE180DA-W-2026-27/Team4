import json
import matplotlib.pyplot as plt
import glob
import os
import sys


files = sorted(glob.glob(os.path.join("maps", "store_map_*.json")))  # oldest -> newest

if len(sys.argv) > 1 and sys.argv[1] == "list":
    for i, fn in enumerate(files, 1):
        with open(fn) as f:
            d = json.load(f)
        print(f"{i}. {os.path.basename(fn)}  ({len(d['path_history']) - 1} steps, {len(d['waypoints'])} waypoints)")
    sys.exit(0)

if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg.isdigit():
        if not 1 <= int(arg) <= len(files):
            print(f"[-] No map number {arg}. Run 'python vis_map.py list' to see them.")
            sys.exit(1)
        MAP_FILE = files[int(arg) - 1]
    else:
        MAP_FILE = arg        
elif files:
    MAP_FILE = files[-1]        
else:
    print("[-] No maps found in the maps folder. Run live_imu_mapper.py first!")
    sys.exit(1)

with open(MAP_FILE, "r") as f:
    map_data = json.load(f)


path = map_data.get("path_history", [])
x_coords = [pt["x"] for pt in path]
y_coords = [pt["y"] for pt in path]

plt.figure(figsize=(6, 6))
plt.plot(
    x_coords,
    y_coords,
    color="red",
    linewidth=3,
    marker="o",
    markersize=5,
    label="Walking Route",
)

if x_coords and y_coords:
  plt.scatter(
      x_coords[0],
      y_coords[0],
      color="green",
      s=100,
      zorder=5,
      label="Start Point",
  )
  plt.scatter(
      x_coords[-1],
      y_coords[-1],
      color="blue",
      s=100,
      zorder=5,
      label="Current/End Position",
  )

plt.title("Assistive Store / Apartment SLAM Map", fontsize=14)
plt.xlabel("X Coordinate (Aisle Width)", fontsize=12)
plt.ylabel("Y Coordinate (Steps / Length)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.axis("equal")

print("[+] Rendering map visualization window...")
plt.show()