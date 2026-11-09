import json
import matplotlib.pyplot as plt

MAP_FILE = "store_map.json"

try:
  with open(MAP_FILE, "r") as f:
    map_data = json.load(f)
except FileNotFoundError:
  print(
      f"[-] Error: {MAP_FILE} not found. Run your imu_mapper script first!"
  )
  exit()

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