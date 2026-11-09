import json, math, time, serial

PORT, BAUD, MAP_FILE = "COM6", 9600, "store_map.json"
STEP_THRESHOLD = 18000
STEP_REFRACTORY = 0.3  
STEP_LENGTH = 1.0       
STALL_TIMEOUT = 2.0

map_data = {"current_position": {"x": 0.0, "y": 0.0},
            "path_history": [{"x": 0.0, "y": 0.0}], "waypoints": []}

def save():
    with open(MAP_FILE, "w") as f:
        json.dump(map_data, f, indent=4)

def parse(line):
    parts = line.split(",")
    if len(parts) != 4:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None

heading = 0.0
step_count = 0
prev_mag = prev_prev_mag = 0.0
last_step_t = 0.0
last_data_t = time.time()
last_t = None
good = bad = 0
max_mag = 0.0
last_report = time.time()

try:
    with serial.Serial(PORT, BAUD, timeout=0.2) as ser:
        time.sleep(3)
        ser.reset_input_buffer()

        print("Calibrating gyro, hold still...")
        samples, t0 = [], time.time()
        while time.time() - t0 < 2.0:
            vals = parse(ser.readline().decode("utf-8", errors="ignore").strip())
            if vals:
                samples.append(vals[0])
        gz_bias = sum(samples) / len(samples) if samples else 0
        print(f"Bias: {gz_bias:.1f} ({len(samples)} samples)")
        print("Walk! Ctrl+C to finish.")

        while True:
            now = time.time()
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            vals = parse(line) if line else None

            if vals is None:
                if line:
                    bad += 1
                if now - last_data_t > STALL_TIMEOUT:
                    print("[!] No data for 2s: check Bluetooth/Arduino")
                    last_data_t = now 
            else:
                good += 1
                last_data_t = now
                gz, ax, ay, az = vals

                dt = (now - last_t) if last_t else 0.0
                last_t = now
                gz_c = gz - gz_bias
                if abs(gz_c) > 50:
                    heading -= math.radians(gz_c * 0.00875)* dt

                mag = math.sqrt(ax**2 + ay**2 + az**2)
                max_mag = max(max_mag, mag)

                if (prev_mag > STEP_THRESHOLD and prev_mag >= prev_prev_mag
                        and prev_mag > mag and now - last_step_t > STEP_REFRACTORY):
                    step_count += 1
                    last_step_t = now
                    map_data["current_position"]["x"] += STEP_LENGTH * math.sin(heading)
                    map_data["current_position"]["y"] += STEP_LENGTH * math.cos(heading)
                    pos = {k: round(v, 2) for k, v in map_data["current_position"].items()}
                    map_data["path_history"].append(pos)
                    save()
                    print(f"[STEP {step_count}] {math.degrees(heading):.0f}° | {pos}")

                prev_prev_mag, prev_mag = prev_mag, mag

            if now - last_report >= 1.0:
                print(f"[rate] good={good}/s bad={bad}/s max_mag={max_mag:.0f}")
                good = bad = 0
                max_mag = 0
                last_report = now

except KeyboardInterrupt:
    save()
    print(f"\nDone. {step_count} steps saved to {MAP_FILE}.")
except serial.SerialException as e:
    save()
    print(f"[-] Serial error (link dropped?): {e}")