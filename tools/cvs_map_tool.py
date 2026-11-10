#!/usr/bin/env python3
"""CVS Store Map CLI Utility.

Interactive command-line tool to build, validate, search, simulate navigation,
and visualize retail store maps for the blind shopping assistant.

Usage:
    python tools/cvs_map_tool.py validate --map data/sample_cvs_map.json
    python tools/cvs_map_tool.py search --query "Where is the moisturizer?" --map data/sample_cvs_map.json
    python tools/cvs_map_tool.py navigate --from entrance --query "moisturizer" --map data/sample_cvs_map.json
    python tools/cvs_map_tool.py visualize --map data/sample_cvs_map.json --svg data/store_map.svg
    python tools/cvs_map_tool.py create --output data/my_store_map.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navigation import (
    MapSerializer,
    ProductIndexer,
    StoreMap,
    StoreMetadata,
    MapNode,
    MapEdge,
    NodeType,
    ProductPlacement,
    ShelfHeight,
    ShelfSide,
    StoreNavigator,
    VisualLandmark,
)


def cmd_validate(args):
    """Validate a map JSON file."""
    print(f"\n[+] Validating store map: {args.map}...")
    issues = MapSerializer.validate_file(args.map)
    if issues:
        print("\n[-] Validation found issues:")
        for issue in issues:
            print(f"  • {issue}")
        sys.exit(1)
    else:
        store_map = MapSerializer.load_from_json(args.map)
        print(f"[✓] Map is valid! '{store_map.metadata.store_name}'")
        print(f"    Nodes: {len(store_map.nodes)}")
        print(f"    Edges: {len(store_map.edges)}")
        total_products = sum(len(n.products) for n in store_map.nodes.values())
        print(f"    Indexed Products: {total_products}")


def cmd_search(args):
    """Search for products using natural language or keywords."""
    store_map = MapSerializer.load_from_json(args.map)
    indexer = ProductIndexer(store_map)

    print(f"\n[+] Searching for: '{args.query}' in {store_map.metadata.store_name}...")
    results = indexer.search(args.query, limit=args.limit)

    if not results:
        print("[-] No matching products found. Try synonyms or broader categories.")
        return

    print(f"\nFound {len(results)} matches:")
    for i, res in enumerate(results, 1):
        print(f"\n  {i}. {res.product.name} (Match Score: {res.match_score:.1f})")
        print(f"     Location: {res.node_name} [{res.node_id}]")
        print(f"     Shelf: {res.shelf_side.value.upper()} side, {res.shelf_height.value.replace('_', ' ').upper()} level")
        if res.product.description:
            print(f"     Details: {res.product.description}")


def cmd_navigate(args):
    """Simulate route planning and generate spoken turn-by-turn directions."""
    store_map = MapSerializer.load_from_json(args.map)
    navigator = StoreNavigator(store_map)
    indexer = ProductIndexer(store_map)

    target_product = None
    goal_node_id = args.to

    if args.query:
        results = indexer.search(args.query, limit=1)
        if not results:
            print(f"[-] Could not find any product matching '{args.query}'.")
            return
        match = results[0]
        goal_node_id = match.node_id
        target_product = match.product
        print(f"[+] Found product '{match.product.name}' at node '{goal_node_id}'")

    if not goal_node_id:
        print("[-] Error: You must specify either --to <node_id> or --query <product_name>")
        return

    plan = navigator.plan_route(
        start_node_id=args.from_node,
        goal_node_id=goal_node_id,
        initial_heading_deg=args.heading,
        target_product=target_product,
    )

    if not plan:
        print(f"[-] No valid path found from '{args.from_node}' to '{goal_node_id}'.")
        return

    print("\n" + "=" * 60)
    print(f"ROUTE PLAN: {plan.destination_name}")
    print(f"Total distance: {plan.total_distance_meters:.1f} m (~{plan.total_steps} steps)")
    print(f"Path Waypoints: {' -> '.join(plan.path_node_ids)}")
    print("=" * 60)
    print("\n[Spoken Turn-by-Turn Audio Guidance for User]:")

    print(f"\n[Departure Overview]:\n  🔊 \"{plan.get_summary_prompt()}\"")

    for step in plan.steps:
        print(f"\nStep #{step.step_index + 1} ({step.action.upper()}):")
        if step.distance_meters > 0:
            print(f"  • Distance: {step.distance_meters:.1f}m ({step.step_count} steps)")
        if step.turn_angle_deg is not None:
            print(f"  • Turn Angle: {step.turn_angle_deg:+.0f}°")
        print(f"  🔊 Spoken: \"{step.verbal_instruction}\"")
        if step.shelf_details:
            print(f"  • Shelf Alignment: {step.shelf_details}")


def cmd_visualize(args):
    """Render ASCII text layout and optionally write SVG map."""
    store_map = MapSerializer.load_from_json(args.map)
    diagram = MapSerializer.generate_ascii_diagram(store_map)
    print("\n" + diagram)

    if args.svg:
        svg_path = Path(args.svg)
        MapSerializer.generate_svg_map(store_map, svg_path)
        print(f"\n[✓] Scalable 2D SVG map saved to: {svg_path.resolve()}")


def cmd_create(args):
    """Interactive wizard to create a new store map JSON from scratch."""
    print("\n=== Interactive CVS Store Map Creator ===")
    store_name = input("Enter Store Name [CVS Pharmacy]: ").strip() or "CVS Pharmacy"
    store_id = input("Enter Store ID [cvs_store_01]: ").strip() or "cvs_store_01"
    address = input("Enter Store Address: ").strip()
    stride_str = input("Average step length in meters [0.70]: ").strip() or "0.70"

    store_map = StoreMap(
        metadata=StoreMetadata(
            store_name=store_name,
            store_id=store_id,
            address=address or None,
            default_step_length_m=float(stride_str),
        )
    )

    print("\n[1] Define Entrance Waypoint:")
    ent_name = input("Entrance name [Store Entrance]: ").strip() or "Store Entrance"
    store_map.nodes["entrance"] = MapNode(
        node_id="entrance",
        name=ent_name,
        node_type=NodeType.ENTRANCE,
        x=0.0,
        y=0.0,
        description="Main front entrance"
    )

    print("\n[2] Add Additional Waypoints & Aisle Nodes (type 'done' when finished):")
    node_count = 1
    while True:
        print("-" * 40)
        nid = input(f"Node ID (e.g. aisle_1_start) or 'done': ").strip()
        if nid.lower() in ("done", "exit", "q", ""):
            break

        name = input(f"Human Name (e.g. Aisle 1 - Oral Care): ").strip()
        ntype_str = input("Type (entrance, intersection, aisle_waypoint, shelf_target, checkout) [aisle_waypoint]: ").strip() or "aisle_waypoint"
        try:
            ntype = NodeType(ntype_str)
        except ValueError:
            ntype = NodeType.AISLE_WAYPOINT

        x_str = input("X coordinate in meters relative to entrance [0.0]: ").strip() or "0.0"
        y_str = input("Y coordinate in meters relative to entrance [5.0]: ").strip() or "5.0"

        # Optional landmark sign
        landmarks = []
        sign_text = input("Overhead Aisle Sign text (optional, press Enter to skip): ").strip()
        if sign_text:
            landmarks.append(VisualLandmark(
                landmark_id=f"lm_{nid}",
                landmark_type="overhead_aisle_sign",
                text=sign_text,
            ))

        # Optional product placement
        products = []
        while True:
            add_prod = input("Add product at this shelf? (y/n) [n]: ").strip().lower()
            if add_prod != "y":
                break
            p_name = input("  Product Name (e.g. CeraVe Moisturizing Cream): ").strip()
            p_cat = input("  Category (e.g. moisturizer): ").strip()
            p_brand = input("  Brand (e.g. CeraVe): ").strip()
            p_side = input("  Shelf Side (left/right/endcap) [right]: ").strip() or "right"
            p_height = input("  Shelf Height (top/eye_level/middle/bottom) [middle]: ").strip() or "middle"
            p_kw = input("  Keywords (comma separated): ").strip()
            kw_list = [k.strip() for k in p_kw.split(",") if k.strip()]

            products.append(ProductPlacement(
                product_id=f"prod_{len(products)+1}_{nid}",
                name=p_name,
                category=p_cat,
                brand=p_brand or None,
                shelf_side=ShelfSide(p_side),
                shelf_height=ShelfHeight(p_height),
                keywords=kw_list,
            ))

        store_map.nodes[nid] = MapNode(
            node_id=nid,
            name=name,
            node_type=ntype,
            x=float(x_str),
            y=float(y_str),
            landmarks=landmarks,
            products=products,
        )
        node_count += 1

    print("\n[3] Connect Waypoints with Navigable Edges (type 'done' when finished):")
    print("Available nodes: " + ", ".join(store_map.nodes.keys()))
    while True:
        print("-" * 40)
        from_n = input("From Node ID (or 'done'): ").strip()
        if from_n.lower() in ("done", "exit", "q", ""):
            break
        to_n = input("To Node ID: ").strip()
        if from_n not in store_map.nodes or to_n not in store_map.nodes:
            print("[-] Error: Node ID not found in map. Try again.")
            continue

        dist_str = input("Distance in meters (or paces * 0.7): ").strip()
        head_str = input("Compass heading in degrees (0=North, 90=East, 180=South, 270=West): ").strip()

        store_map.edges.append(MapEdge(
            from_node=from_n,
            to_node=to_n,
            distance_meters=float(dist_str),
            heading_deg=float(head_str),
            bidirectional=True,
        ))

    output_path = Path(args.output)
    MapSerializer.save_to_json(store_map, output_path)
    print(f"\n[✓] Store map successfully saved to: {output_path.resolve()}")


def cmd_import_imu(args):
    """Convert raw IMU store_map.json (from live_imu_mapper) into a full StoreMap with Dijkstra routing."""
    import json, math, re

    imu_path = Path(args.imu_map)
    if not imu_path.exists():
        print(f"[-] Error: IMU map file '{imu_path}' not found.")
        return

    with open(imu_path, "r") as f:
        data = json.load(f)

    waypoints = data.get("waypoints", [])
    if not waypoints:
        print("[-] Warning: No tagged waypoints found in store_map.json. Tag waypoints during your walk first!")
        return

    print(f"\n[+] Importing {len(waypoints)} waypoints from IMU walk: {imu_path}...")

    store_map = StoreMap(
        metadata=StoreMetadata(
            store_name=args.store_name,
            store_id="cvs_imu_import",
            default_step_length_m=0.7,
        )
    )

    # Always ensure entrance at (0, 0)
    entrance_id = "entrance"
    store_map.nodes[entrance_id] = MapNode(
        node_id=entrance_id,
        name="Store Entrance",
        node_type=NodeType.ENTRANCE,
        x=0.0,
        y=0.0,
        description="Starting location from IMU survey"
    )

    prev_node_id = entrance_id
    prev_x, prev_y = 0.0, 0.0

    for i, wp in enumerate(waypoints, 1):
        raw_name = wp.get("name", f"Section {i}")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw_name).strip("_").lower()
        node_id = f"node_{i}_{slug}"
        side_str = wp.get("side", "center").lower()
        wx = float(wp.get("x", 0.0))
        wy = float(wp.get("y", 0.0))

        products = []
        if side_str in ("right", "left"):
            node_type = NodeType.SHELF_TARGET
            shelf_side = ShelfSide.RIGHT if side_str == "right" else ShelfSide.LEFT
            products.append(
                ProductPlacement(
                    product_id=f"prod_{i}",
                    name=raw_name,
                    category=raw_name.lower(),
                    shelf_side=shelf_side,
                    shelf_height=ShelfHeight.MIDDLE,
                    keywords=[k.strip().lower() for k in raw_name.replace("(", " ").replace(")", " ").split() if len(k) > 2]
                )
            )
        else:
            node_type = NodeType.INTERSECTION

        store_map.nodes[node_id] = MapNode(
            node_id=node_id,
            name=raw_name,
            node_type=node_type,
            x=wx,
            y=wy,
            products=products,
        )

        # Connect with edge
        dx = wx - prev_x
        dy = wy - prev_y
        dist = max(0.5, round(math.hypot(dx, dy), 2))
        # 0 deg = North (+Y), 90 deg = East (+X)
        heading_deg = round(math.degrees(math.atan2(dx, dy)) % 360, 1)

        store_map.edges.append(
            MapEdge(
                from_node=prev_node_id,
                to_node=node_id,
                distance_meters=dist,
                heading_deg=heading_deg,
                bidirectional=True,
                approx_steps=max(1, round(dist / 0.7)),
            )
        )

        prev_node_id = node_id
        prev_x, prev_y = wx, wy

    out_path = Path(args.output)
    MapSerializer.save_to_json(store_map, out_path)
    print(f"[✓] Successfully generated navigable StoreMap: {out_path.resolve()}")
    print(f"    Nodes created: {len(store_map.nodes)}")
    print(f"    Edges created: {len(store_map.edges)}")
    print(f"\nYou can now simulate navigation immediately:")
    print(f"  python tools/cvs_map_tool.py navigate --map {out_path} --from entrance --query 'moisturizer'")


def main():
    parser = argparse.ArgumentParser(
        description="CVS Store Map CLI Tool for Blind Assistive Shopping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Validate
    p_val = subparsers.add_parser("validate", help="Validate store map file")
    p_val.add_argument("--map", default="data/sample_cvs_map.json", help="Path to store map JSON")

    # Search
    p_search = subparsers.add_parser("search", help="Fuzzy search products or categories")
    p_search.add_argument("--query", "-q", required=True, help="Product or question query")
    p_search.add_argument("--map", default="data/sample_cvs_map.json", help="Path to store map JSON")
    p_search.add_argument("--limit", type=int, default=5, help="Max results")

    # Navigate
    p_nav = subparsers.add_parser("navigate", help="Simulate turn-by-turn route and spoken prompts")
    p_nav.add_argument("--from", dest="from_node", default="entrance", help="Starting node ID")
    p_nav.add_argument("--to", dest="to", default=None, help="Destination node ID")
    p_nav.add_argument("--query", "-q", default=None, help="Target product query (e.g. moisturizer)")
    p_nav.add_argument("--heading", type=float, default=0.0, help="Initial heading degrees (0=North)")
    p_nav.add_argument("--map", default="data/sample_cvs_map.json", help="Path to store map JSON")

    # Visualize
    p_vis = subparsers.add_parser("visualize", help="Generate ASCII and SVG visual map diagrams")
    p_vis.add_argument("--map", default="data/sample_cvs_map.json", help="Path to store map JSON")
    p_vis.add_argument("--svg", default=None, help="Optional output path for SVG file")

    # Create
    p_create = subparsers.add_parser("create", help="Interactive store map builder wizard")
    p_create.add_argument("--output", "-o", default="data/my_cvs_map.json", help="Output file path")

    # Import IMU
    p_imp = subparsers.add_parser("import-imu", help="Convert raw IMU store_map.json into navigable StoreMap")
    p_imp.add_argument("--imu-map", default="imu_Test/store_map.json", help="Path to raw IMU store_map.json")
    p_imp.add_argument("--output", "-o", default="data/cvs_from_imu.json", help="Output StoreMap JSON path")
    p_imp.add_argument("--store-name", default="CVS Pharmacy (IMU Survey)", help="Store name")

    args = parser.parse_args()

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "navigate":
        cmd_navigate(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "create":
        cmd_create(args)
    elif args.command == "import-imu":
        cmd_import_imu(args)


if __name__ == "__main__":
    main()

