import argparse
import glob
import json
import math
import os
import sys
import threading
import time
import serial

DEFAULT_BAUD = 9600
MAP_FILE = "store_map.json"
STEP_THRESHOLD = 18000
STEP_REFRACTORY = 0.3
STEP_LENGTH = 0.7  
STALL_TIMEOUT = 2.0

map_data = {
    "current_position": {"x": 0.0, "y": 0.0},
    "path_history": [{"x": 0.0, "y": 0.0}],
    "waypoints": []
}

data_lock = threading.RLock()
running = True

# Shared live state
heading = 0.0
step_count = 0
last_step_pos = {"x": 0.0, "y": 0.0}


def auto_detect_port():
    """Auto-detect available serial port across Mac, Linux, and Windows."""
    if sys.platform.startswith("darwin"):
        ports = glob.glob("/dev/cu.usb*") + glob.glob("/dev/cu.*HC*") + glob.glob("/dev/cu.*DSD*")
        if ports:
            return ports[0]
        all_cu = [p for p in glob.glob("/dev/cu.*") if "Bluetooth" not in p and "debug" not in p]
        if all_cu:
            return all_cu[0]
    elif sys.platform.startswith("linux"):
        ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/rfcomm*")
        if ports:
            return ports[0]
    elif sys.platform.startswith("win"):
        return "COM6"
    return "/dev/cu.usbmodem14101" if sys.platform.startswith("darwin") else "COM6"


def save():
    """Thread-safe save to store_map.json."""
    with data_lock:
        with open(MAP_FILE, "w") as f:
            json.dump(map_data, f, indent=4)


def add_waypoint(name: str, side: str = "right"):
    """Drop a tagged category/section waypoint at the current live position."""
    global step_count, heading
    with data_lock:
        curr_x = round(map_data["current_position"]["x"], 2)
        curr_y = round(map_data["current_position"]["y"], 2)
        deg = round(math.degrees(heading) % 360, 1)

        # Normalize side
        side_clean = side.lower().strip()
        if side_clean in ("r", "right"):
            side_val = "right"
        elif side_clean in ("l", "left"):
            side_val = "left"
        else:
            side_val = "center"

        wp = {
            "name": name.strip(),
            "side": side_val,
            "x": curr_x,
            "y": curr_y,
            "heading_deg": deg,
            "step": step_count,
            "time": time.strftime("%H:%M:%S")
        }
        map_data["waypoints"].append(wp)
        with open(MAP_FILE, "w") as f:
            json.dump(map_data, f, indent=4)

    side_label = f" ({side_val.upper()} side)" if side_val != "center" else " (CORRIDOR/JUNCTION)"
    print(f"\n[★ WAYPOINT ADDED] '{name}'{side_label} at step {step_count} -> ({curr_x}, {curr_y})\n", flush=True)


def waypoint_cli_thread():
    """Background console listener allowing user to type section names in real-time without blocking serial."""
    global running
    print("\n" + "=" * 60)
    print("  INTERACTIVE WAYPOINT DROPPER READY")
    print("  • Type section name to drop pin: e.g. 'Moisturizers - r'")
    print("  • Format: <Name> [- r | - l | - c]  (defaults to right)")
    print("  • Examples:")
    print("      Moisturizers - r   -> Moisturizers on the right shelf")
    print("      Sunscreen - l      -> Sunscreen on the left shelf")
    print("      Aisle 4 Entrance   -> Aisle intersection (center)")
    print("  • Type 'list' to view waypoints, 'undo' to remove last")
    print("=" * 60 + "\n", flush=True)

    while running:
        try:
            cmd = input().strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd or not running:
            continue

        if cmd.lower() == "list":
            with data_lock:
                wps = map_data.get("waypoints", [])
            print(f"\n--- Current Waypoints ({len(wps)}) ---")
            for i, w in enumerate(wps, 1):
                print(f"  {i}. {w['name']} [{w['side'].upper()}] @ ({w['x']}, {w['y']}) - Step {w['step']}")
            print("------------------------------\n", flush=True)
            continue

        if cmd.lower() == "undo":
            with data_lock:
                if map_data.get("waypoints"):
                    removed = map_data["waypoints"].pop()
                    save()
                    print(f"[!] Removed last waypoint: {removed['name']}\n", flush=True)
                else:
                    print("[!] No waypoints to undo.\n", flush=True)
            continue

        side = "right"  # default
        name = cmd

        if " - " in cmd:
            parts = cmd.split(" - ", 1)
            name = parts[0].strip()
            side = parts[1].strip()
        elif cmd.lower().endswith(" -r") or cmd.lower().endswith(" -l"):
            name = cmd[:-3].strip()
            side = cmd[-1].strip()
        elif cmd.lower().endswith(" right"):
            name = cmd[:-6].strip()
            side = "right"
        elif cmd.lower().endswith(" left"):
            name = cmd[:-5].strip()
            side = "left"

        add_waypoint(name, side)


def parse(line):
    parts = line.split(",")
    if len(parts) != 4:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def main():
    global heading, step_count, running

    parser = argparse.ArgumentParser(description="Live IMU PDR Mapper with Interactive Category Waypoint Tagger")
    parser.add_argument("--port", "-p", default=None, help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baud", "-b", type=int, default=DEFAULT_BAUD, help=f"Baud rate (default: {DEFAULT_BAUD})")
    parser.add_argument("--step-length", type=float, default=STEP_LENGTH, help=f"Step length in meters (default: {STEP_LENGTH})")
    args = parser.parse_args()

    port = args.port or auto_detect_port()
    baud = args.baud
    step_len = args.step_length

    print(f"[+] Using Serial Port: {port} @ {baud} baud (Step length: {step_len}m)")

    prev_mag = prev_prev_mag = 0.0
    last_step_t = 0.0
    last_data_t = time.time()
    last_t = None
    good = bad = 0
    max_mag = 0.0
    last_report = time.time()

    try:
        ser = serial.Serial(port, baud, timeout=0.2)
    except serial.SerialException as e:
        print(f"\n[-] Could not open serial port '{port}': {e}")
        print("\nAvailable ports on your system:")
        if sys.platform.startswith("darwin"):
            for p in glob.glob("/dev/cu.*"):
                print(f"  • {p}")
        print("\nPass your port explicitly: python live_imu_mapper.py --port <PORT_PATH>")
        sys.exit(1)

    time.sleep(3)
    ser.reset_input_buffer()

    print("Calibrating gyro, hold still...")
    samples, t0 = [], time.time()
    while time.time() - t0 < 2.0:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        vals = parse(line) if line else None
        if vals:
            samples.append(vals[0])
    gz_bias = sum(samples) / len(samples) if samples else 0
    print(f"Bias: {gz_bias:.1f} ({len(samples)} samples)")

    # Start interactive waypoint dropper in background thread
    t = threading.Thread(target=waypoint_cli_thread, daemon=True)
    t.start()

    print("Walk! Type section name anytime to drop a waypoint. Ctrl+C to finish.")

    try:
        while True:
            now = time.time()
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            vals = parse(line) if line else None

            if vals is None:
                if line:
                    bad += 1
                if now - last_data_t > STALL_TIMEOUT:
                    print("[!] No data for 2s: check Bluetooth/Arduino", flush=True)
                    last_data_t = now
            else:
                good += 1
                last_data_t = now
                gz, ax, ay, az = vals

                dt = (now - last_t) if last_t else 0.0
                last_t = now
                gz_c = gz - gz_bias
                rate_dps = abs(gz_c) * 0.00875
                if abs(gz_c) > 50:
                    heading -= math.radians(gz_c * 0.00875) * dt

                mag = math.sqrt(ax**2 + ay**2 + az**2)
                max_mag = max(max_mag, mag)

                if (prev_mag > STEP_THRESHOLD and prev_mag >= prev_prev_mag
                        and prev_mag > mag and now - last_step_t > STEP_REFRACTORY):
                    step_count += 1
                    last_step_t = now
                    with data_lock:
                        map_data["current_position"]["x"] += step_len * math.sin(heading)
                        map_data["current_position"]["y"] += step_len * math.cos(heading)
                        pos = {k: round(v, 2) for k, v in map_data["current_position"].items()}
                        map_data["path_history"].append(pos)
                        save()
                    print(f"[STEP {step_count}] {math.degrees(heading):.0f}° | {pos}", flush=True)

                prev_prev_mag, prev_mag = prev_mag, mag

            if now - last_report >= 1.0:
                # Keep rate line minimal so it doesn't clutter user typing
                if bad > 0:
                    print(f"[rate] good={good}/s bad={bad}/s max_mag={max_mag:.0f}", flush=True)
                good = bad = 0
                max_mag = 0
                last_report = now

    except KeyboardInterrupt:
        running = False
        save()
        with data_lock:
            n_wps = len(map_data.get("waypoints", []))
        print(f"\nDone. {step_count} steps and {n_wps} waypoints saved to {MAP_FILE}.")
    except serial.SerialException as e:
        running = False
        save()
        print(f"[-] Serial error (link dropped?): {e}")


if __name__ == "__main__":
    main()
