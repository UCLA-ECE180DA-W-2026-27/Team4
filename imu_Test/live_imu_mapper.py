import json
import math
import os
import time
import serial

PORT = "COM6"
BAUD = 9600
MAP_FILE = "store_map.json"

# Initialize map database
map_data = {
    "current_position": {"x": 0.0, "y": 0.0},
    "path_history": [{"x": 0.0, "y": 0.0}],
    "waypoints": [],
}
with open(MAP_FILE, "w") as f:
  json.dump(map_data, f, indent=4)

print(f"Connecting to Walking SLAM Mapper on {PORT}...")

# Step detection variables
last_magnitude = 0.0
step_threshold = 18000  # Adjust this based on how hard the sensor bounces when walked with
cooldown = 0
step_count = 0

try:
  with serial.Serial(PORT, BAUD, timeout=1) as ser:
    print("[+] Connected! Waiting for Arduino reboot...")
    time.sleep(3)
    print(
        "[+] Walking SLAM Active. Walk with your sensor (Ctrl+C to finish &"
        " save)..."
    )

    while True:
      line = ser.readline().decode("utf-8", errors="ignore").strip()
      if line and "," in line:
        try:
          parts = line.split(",")
          if len(parts) == 3:
            x = int(parts[0])
            y = int(parts[1])
            z = int(parts[2])

            # Calculate 3D acceleration magnitude (Vector length)
            magnitude = math.sqrt(x**2 + y**2 + z**2)

            # Simple Peak Detection for Step Counting (PDR)
            # If magnitude spikes above threshold and cooldown has passed
            if (
                magnitude > step_threshold
                and magnitude > last_magnitude
                and cooldown == 0
            ):
              step_count += 1
              # Assuming you are walking "forward" along the Y-axis (aisle mapping)
              map_data["current_position"]["y"] += 1.0

              current_pos = {
                  "x": round(map_data["current_position"]["x"], 2),
                  "y": round(map_data["current_position"]["y"], 2),
              }
              map_data["path_history"].append(current_pos)

              # Write live to store_map.json
              with open(MAP_FILE, "w") as f:
                json.dump(map_data, f, indent=4)

              print(
                  f"[STEP {step_count}] Magnitude: {int(magnitude)} | Position"
                  f" -> X: {current_pos['x']}, Y: {current_pos['y']}"
              )

              cooldown = 8  # Prevent multiple triggers for a single footstep (debounce)

            if cooldown > 0:
              cooldown -= 1

            last_magnitude = magnitude

        except ValueError:
          pass

except KeyboardInterrupt:
  print(
      f"\n[!] Mapping session ended. Total steps logged: {step_count}. Map"
      f" saved to '{MAP_FILE}'."
  )
except Exception as e:
  print(f"[-] Error: {e}")