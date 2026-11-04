"""Serialization, persistence, validation, and visual rendering of Store Maps.

Provides utilities to read/write JSON files, generate ASCII layout diagrams,
and export scalable visual SVG maps of the retail environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from navigation.map_schema import StoreMap
from navigation.store_graph import StoreGraph


class MapSerializer:
    """Handles file I/O, validation, and rendering for StoreMap objects."""

    @staticmethod
    def load_from_json(file_path: str | Path) -> StoreMap:
        """Load and validate a StoreMap from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Store map file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return StoreMap.model_validate(data)

    @staticmethod
    def save_to_json(store_map: StoreMap, file_path: str | Path, indent: int = 2) -> None:
        """Serialize and save a StoreMap to formatted JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = store_map.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)

    @staticmethod
    def validate_file(file_path: str | Path) -> List[str]:
        """Validate a store map file on disk and return any structural issues."""
        try:
            store_map = MapSerializer.load_from_json(file_path)
            graph = StoreGraph(store_map)
            return graph.validate_graph()
        except Exception as e:
            return [f"Failed to parse or validate schema: {e}"]

    @staticmethod
    def generate_ascii_diagram(store_map: StoreMap) -> str:
        """Render a clean text-based directory and connectivity summary."""
        lines = [
            f"=== {store_map.metadata.store_name} Map Layout ===",
            f"Store ID: {store_map.metadata.store_id} | Version: {store_map.metadata.version}",
            f"Total Waypoints/Nodes: {len(store_map.nodes)} | Edges: {len(store_map.edges)}",
            "",
            "--- Waypoints & Product Shelves ---",
        ]

        for node_id, node in sorted(store_map.nodes.items()):
            prods = [f"{p.name} ({p.shelf_side.value}, {p.shelf_height.value})" for p in node.products]
            prod_str = f" | Products: {', '.join(prods)}" if prods else ""
            signs = [s.text for s in node.landmarks if s.text]
            sign_str = f" | Signs: [{', '.join(signs)}]" if signs else ""
            lines.append(f"  [{node.node_type.value.upper()}] {node_id} ({node.x:.1f}m, {node.y:.1f}m): {node.name}{sign_str}{prod_str}")

        lines.append("")
        lines.append("--- Connectivity & Distances ---")
        for edge in store_map.edges:
            bi_arrow = "<--->" if edge.bidirectional else "--->"
            lines.append(f"  {edge.from_node} {bi_arrow} {edge.to_node} : {edge.distance_meters:.1f}m @ {edge.heading_deg:.0f}°")

        return "\n".join(lines)

    @staticmethod
    def generate_svg_map(store_map: StoreMap, output_file: Optional[str | Path] = None) -> str:
        """Generate a visual 2D SVG map diagram of the store with aisles, nodes, and shelves."""
        if not store_map.nodes:
            return "<svg></svg>"

        # Find coordinate bounds
        xs = [n.x for n in store_map.nodes.values()]
        ys = [n.y for n in store_map.nodes.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        padding = 40.0
        scale = 35.0  # pixels per meter
        width_m = max(1.0, (max_x - min_x))
        height_m = max(1.0, (max_y - min_y))

        svg_width = int(width_m * scale + 2 * padding)
        svg_height = int(height_m * scale + 2 * padding)

        def to_svg_x(x: float) -> float:
            return padding + (x - min_x) * scale

        def to_svg_y(y: float) -> float:
            # Invert Y so North is Up
            return svg_height - (padding + (y - min_y) * scale)

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}" style="background-color: #1e1e2e; font-family: sans-serif;">',
            f'<text x="20" y="30" fill="#cdd6f4" font-size="18" font-weight="bold">{store_map.metadata.store_name} — Store Navigation Map</text>',
            f'<text x="20" y="50" fill="#a6adc8" font-size="12">Scale: {scale}px = 1m | North = Up ↑</text>',
        ]

        # Draw edges (pathways)
        for edge in store_map.edges:
            fn = store_map.nodes.get(edge.from_node)
            tn = store_map.nodes.get(edge.to_node)
            if fn and tn:
                x1, y1 = to_svg_x(fn.x), to_svg_y(fn.y)
                x2, y2 = to_svg_x(tn.x), to_svg_y(tn.y)
                svg_parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#45475a" stroke-width="4" stroke-dasharray="2,2"/>')

        # Draw nodes
        for node_id, node in store_map.nodes.items():
            cx = to_svg_x(node.x)
            cy = to_svg_y(node.y)

            # Color coding by node type
            if node.node_type.value == "entrance":
                color = "#a6e3a1"  # green
                radius = 10
            elif node.node_type.value == "shelf_target":
                color = "#f38ba8"  # red/pink
                radius = 8
            elif node.node_type.value == "intersection":
                color = "#89b4fa"  # blue
                radius = 6
            else:
                color = "#fab387"  # orange
                radius = 6

            svg_parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{color}" stroke="#11111b" stroke-width="2"/>')
            # Label
            label = node.name.split("-")[0].strip()
            svg_parts.append(f'<text x="{cx + 10:.1f}" y="{cy + 4:.1f}" fill="#cdd6f4" font-size="11" font-weight="500">{label}</text>')

            # Show product count badge if shelf
            if node.products:
                p_text = f"{len(node.products)} items"
                svg_parts.append(f'<text x="{cx + 10:.1f}" y="{cy + 16:.1f}" fill="#f9e2af" font-size="9">({p_text})</text>')

        svg_parts.append('</svg>')
        svg_content = "\n".join(svg_parts)

        if output_file:
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(svg_content)

        return svg_content

