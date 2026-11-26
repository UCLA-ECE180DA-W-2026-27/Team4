
import argparse
import glob
import heapq
import json
import math
import os
import sys
import time

DEFAULT_BAUD = 9600
STEP_THRESHOLD = 18000
STEP_REFRACTORY = 0.3
STEP_LENGTH = 0.7
TURN_RATE_LIMIT = 45.0     
STALL_TIMEOUT = 2.0
CALIBRATION_SECONDS = 3.0

MIN_TURN_DEG = 30          
WARN_STEPS = 2              
ARRIVE_M = 1.0              
OFF_ROUTE_M = 1.5          
SHORTCUT_M = 0.8           
RDP_TOLERANCE_M = 0.8      
ALIGN_DEG = 20             


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bearing(a, b):
    return math.atan2(b[0] - a[0], b[1] - a[1])


def ang_diff(a, b):
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def point_line_dist(p, a, b):
    if a == b:
        return dist(p, a)
    return abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1])) / dist(a, b)


def rdp(points, eps):
    if len(points) < 3:
        return list(points)
    a, b = points[0], points[-1]
    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        d = point_line_dist(points[i], a, b)
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        return rdp(points[: idx + 1], eps)[:-1] + rdp(points[idx:], eps)
    return [a, b]


def simplify_route(points):
    corners = rdp(points, RDP_TOLERANCE_M)
    changed = True
    while changed and len(corners) > 2:
        changed = False
        for i in range(1, len(corners) - 1):
            turn = abs(math.degrees(ang_diff(bearing(corners[i - 1], corners[i]),
                                             bearing(corners[i], corners[i + 1]))))
            if turn < MIN_TURN_DEG:
                del corners[i]
                changed = True
                break
    return corners



def build_graph(pts):
    
    n = len(pts)
    adj = {i: [] for i in range(n)}
    for i in range(n - 1):
        d = dist(pts[i], pts[i + 1])
        adj[i].append((i + 1, d))
        adj[i + 1].append((i, d))
    for i in range(n):
        for j in range(i + 4, n):
            d = dist(pts[i], pts[j])
            if d < SHORTCUT_M:
                adj[i].append((j, d))
                adj[j].append((i, d))
    return adj


def shortest_path(adj, s, t):
    best = {s: 0.0}
    prev = {}
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t:
            break
        if d > best.get(u, float("inf")):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < best.get(v, float("inf")):
                best[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if t not in best:
        return None
    path = [t]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return path[::-1]


def approx_deg(rad):
    return int(round(abs(math.degrees(rad)) / 15.0) * 15)


def turn_words(diff):
    side = "right" if diff > 0 else "left"
    a = abs(math.degrees(diff))
    if a >= 135:
        return "turn around"
    if a >= 60:
        return f"turn {side}"
    return f"turn slightly {side}"


def steps_for(length_m, step_len):
    return max(1, int(round(length_m / step_len)))


def side_on_arrival(wp, survey_dir, arrive_dir):
    
    side = wp.get("side", "center")
    if side in ("left", "right") and abs(ang_diff(survey_dir, arrive_dir)) > math.pi / 2:
        side = "left" if side == "right" else "right"
    return side


class Navigator:
    def __init__(self, pts, waypoints, step_len):
        self.pts = [tuple(p) for p in pts]
        self.wps = waypoints
        self.step_len = step_len
        self.adj = build_graph(self.pts)
        self.pos = self.pts[0]
        self.heading = 0.0
        self.dest = None
        self.legs = []
        self.k = 0
        self.state = "idle"       
        self.target = 0.0
        self.warned = False
        self.dest_side = "center"


    def nearest_node(self, p):
        return min(range(len(self.pts)), key=lambda i: dist(self.pts[i], p))

    def survey_dir(self, i):
        a = self.pts[max(i - 2, 0)]
        b = self.pts[min(i + 2, len(self.pts) - 1)]
        return bearing(a, b)

    def route_length(self, path):
        return sum(dist(self.pts[path[i]], self.pts[path[i + 1]]) for i in range(len(path) - 1))

    def find(self, query):
        q = query.strip().lower()
        matches = [w for w in self.wps if w["name"].lower() == q]
        if not matches:
            matches = [w for w in self.wps if q in w["name"].lower()]
        if not matches:
            return None
        here = self.nearest_node(self.pos)

        def cost(w):
            p = shortest_path(self.adj, here, self.nearest_node((w["x"], w["y"])))
            return self.route_length(p) if p else float("inf")

        return min(matches, key=cost)   

    def plan(self, wp):
        self.dest = wp
        s = self.nearest_node(self.pos)
        t = self.nearest_node((wp["x"], wp["y"]))
        path = shortest_path(self.adj, s, t)
        if path is None:
            print(f"[!] No walkable route to '{wp['name']}' on this map.", flush=True)
            self.state = "idle"
            self.legs = []
            return
        total = self.route_length(path)
        route = [self.pts[i] for i in path]
        corners = simplify_route(route)
        self.legs = [
            dict(start=corners[i], end=corners[i + 1],
                 heading=bearing(corners[i], corners[i + 1]),
                 length=dist(corners[i], corners[i + 1]))
            for i in range(len(corners) - 1)
        ]
        self.k = 0
        self.warned = False
        if not self.legs or total <= ARRIVE_M:
            self.arrive()
            return
        arrive_dir = self.legs[-1]["heading"]
        self.dest_side = side_on_arrival(wp, self.survey_dir(t), arrive_dir)
        self.print_overview(total)
        self.begin_first_leg()

    def print_overview(self, total):
        name = self.dest["name"]
        print(f"\nRoute to '{name}': about {total:.1f} m ({steps_for(total, self.step_len)} steps)", flush=True)
        for i, leg in enumerate(self.legs):
            walk = f"walk {steps_for(leg['length'], self.step_len)} steps"
            if i == 0:
                print(f"  1. {walk[0].upper() + walk[1:]} straight", flush=True)
            else:
                diff = ang_diff(self.legs[i - 1]["heading"], leg["heading"])
                print(f"  {i + 1}. {turn_words(diff).capitalize()} (about {approx_deg(diff)}°), {walk}", flush=True)
        print(f"  {len(self.legs) + 1}. '{name}' will be on your {self.dest_side.upper()}\n", flush=True)

    def begin_first_leg(self):
        leg = self.legs[0]
        diff = ang_diff(self.heading, leg["heading"])
        if abs(math.degrees(diff)) > MIN_TURN_DEG:
            self.state = "turning"
            self.target = leg["heading"]
            print(f">>> {turn_words(diff).upper()} (about {approx_deg(diff)}°) to face the route", flush=True)
        else:
            self.state = "walking"
            print(f">>> Walk straight {steps_for(leg['length'], self.step_len)} steps", flush=True)

    def project(self, leg):
        h = leg["heading"]
        dx, dy = self.pos[0] - leg["start"][0], self.pos[1] - leg["start"][1]
        along = dx * math.sin(h) + dy * math.cos(h)
        cross = dx * math.cos(h) - dy * math.sin(h)   # + = right of the route
        return along, cross

    def update_heading(self, h):
        self.heading = h
        if self.state == "turning" and abs(math.degrees(ang_diff(h, self.target))) < ALIGN_DEG:
            self.state = "walking"
            self.warned = False
            leg = self.legs[self.k]
            along, _ = self.project(leg)
            remaining = max(leg["length"] - along, 0.0)
            print(f">>> Good. Walk straight {steps_for(remaining, self.step_len)} steps", flush=True)

    def step(self):
        h = self.heading
        self.pos = (self.pos[0] + self.step_len * math.sin(h),
                    self.pos[1] + self.step_len * math.cos(h))
        if self.state != "walking" or not self.legs:
            return
        leg = self.legs[self.k]
        along, cross = self.project(leg)
        remaining = leg["length"] - along

        if abs(cross) > OFF_ROUTE_M or remaining < -OFF_ROUTE_M:
            print("[!] You've moved off the route. Re-planning...", flush=True)
            self.plan(self.dest)
            return

        last = self.k == len(self.legs) - 1
        if last:
            if remaining <= ARRIVE_M:
                self.arrive()
            elif remaining <= WARN_STEPS * self.step_len and not self.warned:
                self.warned = True
                print(f"    Almost there: '{self.dest['name']}' is about "
                      f"{steps_for(remaining, self.step_len)} steps ahead, on your {self.dest_side.upper()}", flush=True)
            return

        nxt = self.legs[self.k + 1]
        diff = ang_diff(leg["heading"], nxt["heading"])
        if remaining <= 0.7 * self.step_len:
            self.k += 1
            self.state = "turning"
            self.target = nxt["heading"]
            self.warned = False
            print(f">>> {turn_words(diff).upper()} now (about {approx_deg(diff)}°)", flush=True)
        elif remaining <= WARN_STEPS * self.step_len and not self.warned:
            self.warned = True
            print(f"    In {steps_for(remaining, self.step_len)} step(s): {turn_words(diff)}", flush=True)

    def arrive(self):
        wp = self.dest
        side = self.dest_side if self.legs else wp.get("side", "center")
        if side == "center":
            where = "at this junction"
        else:
            where = f"on your {side.upper()}"
        print(f"\n*** ARRIVED: '{wp['name']}' is {where} ***\n", flush=True)
        self.pos = (wp["x"], wp["y"])     
        self.state = "idle"
        self.legs = []


def load_map(path_arg):
    if path_arg:
        path = path_arg
    else:
        files = sorted(glob.glob(os.path.join("maps", "store_map_*.json")))
        path = files[-1] if files else "store_map.json"
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[-] Map file not found: {path}. Use --map <file>.")
        sys.exit(1)
    pts = [(p["x"], p["y"]) for p in data.get("path_history", [])]
    wps = data.get("waypoints", [])
    if len(pts) < 2:
        print("[-] This map has no walked path. Survey it with live_imu_mapper.py first.")
        sys.exit(1)
    if not wps:
        print("[-] This map has no waypoints. Tag some while surveying (e.g. 'coffee - r').")
        sys.exit(1)
    print(f"[+] Loaded {path}: {len(pts) - 1} steps, {len(wps)} waypoints")
    return pts, wps


def choose_destination(nav):
    names = sorted({w["name"] for w in nav.wps})
    print("\nWaypoints on this map: " + ", ".join(names))
    while True:
        q = input("Where to? (type a name, or press Enter to quit): ").strip()
        if not q:
            return None
        wp = nav.find(q)
        if wp:
            return wp
        print(f"[!] No waypoint matches '{q}'.")


def parse(line):
    parts = line.split(",")
    if len(parts) != 4:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def run_simulated(nav, dest):
    print("\nSIMULATION: Enter = take a step, l = turn left 90°, r = turn right 90°, q = quit")
    nav.plan(dest)
    while True:
        if nav.state == "idle":
            dest = choose_destination(nav)
            if dest is None:
                return
            nav.plan(dest)
            continue
        cmd = input("> ").strip().lower()
        if cmd == "q":
            return
        if cmd in ("l", "r"):
            nav.update_heading(nav.heading + (math.pi / 2 if cmd == "r" else -math.pi / 2))
        else:
            nav.step()


def run_live(nav, dest, port, baud):
    import serial   

    try:
        ser = serial.Serial(port, baud, timeout=0.2)
    except serial.SerialException as e:
        print(f"[-] Could not open serial port '{port}': {e}")
        sys.exit(1)

    time.sleep(3)
    ser.reset_input_buffer()
    print(f"Stand at your start point, facing the surveyed direction, and hold still "
          f"({CALIBRATION_SECONDS:.0f} s calibration)...")
    samples, t0 = [], time.time()
    while time.time() - t0 < CALIBRATION_SECONDS:
        vals = parse(ser.readline().decode("utf-8", errors="ignore").strip())
        if vals:
            samples.append(vals[0])
    gz_bias = sum(samples) / len(samples) if samples else 0.0
    print(f"Bias: {gz_bias:.1f} ({len(samples)} samples). Go!")

    nav.plan(dest)
    prev_mag = prev_prev_mag = 0.0
    last_step_t = 0.0
    last_data_t = time.time()
    last_t = None

    try:
        while True:
            if nav.state == "idle":
                dest = choose_destination(nav)
                if dest is None:
                    break
                ser.reset_input_buffer()
                last_t = None
                prev_mag = prev_prev_mag = 0.0
                nav.plan(dest)
                continue

            now = time.time()
            vals = parse(ser.readline().decode("utf-8", errors="ignore").strip())
            if vals is None:
                if now - last_data_t > STALL_TIMEOUT:
                    print("[!] No data for 2s: check Bluetooth/Arduino", flush=True)
                    last_data_t = now
                continue
            last_data_t = now
            gz, ax, ay, az = vals

            dt = (now - last_t) if last_t else 0.0
            last_t = now
            gz_c = gz - gz_bias
            rate_dps = abs(gz_c) * 0.00875
            h = nav.heading
            if abs(gz_c) > 50:
                h -= math.radians(gz_c * 0.00875) * dt
            nav.update_heading(h)

            mag = math.sqrt(ax ** 2 + ay ** 2 + az ** 2)
            is_peak = (prev_mag > STEP_THRESHOLD and prev_mag >= prev_prev_mag
                       and prev_mag > mag and now - last_step_t > STEP_REFRACTORY)
            if is_peak and rate_dps < TURN_RATE_LIMIT:
                last_step_t = now
                nav.step()
            prev_prev_mag, prev_mag = prev_mag, mag
    except KeyboardInterrupt:
        pass
    print("\nNavigation ended.")


def main():
    parser = argparse.ArgumentParser(description="Guide the user to a named waypoint using a saved map + IMU")
    parser.add_argument("--map", "-m", default=None, help="Map JSON (default: newest in maps/)")
    parser.add_argument("--port", "-p", default="COM6", help="Serial port (default COM6)")
    parser.add_argument("--baud", "-b", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--step-length", type=float, default=STEP_LENGTH)
    parser.add_argument("--start", default=None, help="Start standing at this waypoint instead of the map start")
    parser.add_argument("--simulate", action="store_true", help="No hardware: Enter = step, l/r = turn")
    args = parser.parse_args()

    pts, wps = load_map(args.map)
    nav = Navigator(pts, wps, args.step_length)

    if args.start:
        w = nav.find(args.start)
        if not w:
            print(f"[-] --start: no waypoint matches '{args.start}'.")
            sys.exit(1)
        nav.pos = (w["x"], w["y"])
        nav.heading = nav.survey_dir(nav.nearest_node(nav.pos))
        print(f"[+] Starting at '{w['name']}', facing the way the map was surveyed there.")
    else:
        print("[+] Starting at the map's start point, facing the surveyed direction.")

    dest = choose_destination(nav)
    if dest is None:
        return
    if args.simulate:
        run_simulated(nav, dest)
    else:
        run_live(nav, dest, args.port, args.baud)


if __name__ == "__main__":
    main()
