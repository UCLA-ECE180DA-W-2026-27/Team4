import json
from pathlib import Path
import matplotlib.pyplot as plt

MAP_FILE = "store_map.json"

map_path = Path(MAP_FILE)
if not map_path.exists():
    # Also check parent / current directory
    alt_path = Path(__file__).parent / "store_map.json"
    if alt_path.exists():
        map_path = alt_path
    else:
        print(f"[-] Error: '{MAP_FILE}' not found. Run live_imu_mapper.py first!")
        exit(1)

with open(map_path, "r") as f:
    map_data = json.load(f)

path = map_data.get("path_history", [])
waypoints = map_data.get("waypoints", [])

x_coords = [pt["x"] for pt in path]
y_coords = [pt["y"] for pt in path]

fig, ax = plt.subplots(figsize=(8, 8))

# 1. Plot continuous walking route
if x_coords and y_coords:
    ax.plot(
        x_coords,
        y_coords,
        color="#e63946",
        linewidth=2.5,
        marker="o",
        markersize=4,
        alpha=0.85,
        label="Walking Trail",
        zorder=2,
    )

    # Start Point
    ax.scatter(
        x_coords[0],
        y_coords[0],
        color="#2a9d8f",
        s=130,
        zorder=5,
        label="Start / Entrance",
        edgecolors="black",
        linewidths=1.5,
    )

    # Current / End Position
    ax.scatter(
        x_coords[-1],
        y_coords[-1],
        color="#457b9d",
        s=130,
        zorder=5,
        label="Current Position",
        edgecolors="black",
        linewidths=1.5,
    )

# 2. Plot Tagged Category Waypoints
if waypoints:
    print("\n" + "=" * 50)
    print(f"  TAGGED STORE SECTIONS & WAYPOINTS ({len(waypoints)})")
    print("=" * 50)
    print(f"{'#':<3} | {'Section Name':<22} | {'Side':<7} | {'Coordinates':<14} | {'Step'}")
    print("-" * 50)

    for i, wp in enumerate(waypoints, 1):
        wx, wy = wp["x"], wp["y"]
        wname = wp.get("name", f"Waypoint {i}")
        wside = wp.get("side", "center").upper()
        wstep = wp.get("step", "-")

        print(f"{i:<3} | {wname:<22} | {wside:<7} | ({wx:>5.1f}, {wy:>5.1f})   | {wstep}")

        # Choose marker color based on side
        marker_color = "#e76f51" if wside == "RIGHT" else ("#f4a261" if wside == "LEFT" else "#e9c46a")

        # Scatter waypoint star
        ax.scatter(
            wx,
            wy,
            color=marker_color,
            s=220,
            marker="*",
            zorder=6,
            edgecolors="#1d3557",
            linewidths=1.2,
            label="Tagged Section" if i == 1 else "",
        )

        # Annotate with text callout box
        side_tag = f" [{wside[0]}]" if wside in ("RIGHT", "LEFT") else ""
        label_text = f"{wname}{side_tag}"

        ax.annotate(
            label_text,
            (wx, wy),
            textcoords="offset points",
            xytext=(10, 10 if i % 2 == 0 else -15),
            fontsize=9,
            fontweight="bold",
            color="#1d3557",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f1faee", ec=marker_color, lw=1.5, alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color=marker_color, lw=1.2),
            zorder=7,
        )
    print("-" * 50 + "\n")
else:
    print("[i] No category waypoints tagged yet in this map.")

ax.set_title("CVS Store PDR Map with Category Waypoints", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("X Distance (Meters - East / Aisle Offset)", fontsize=11)
ax.set_ylabel("Y Distance (Meters - North / Corridor Depth)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left", framealpha=0.9)
ax.axis("equal")

print("[+] Rendering map visualization window...")
plt.tight_layout()
plt.show()
