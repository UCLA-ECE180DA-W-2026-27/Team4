"""Store graph topology, pathfinding, and geometry engine.

Builds an adjacency graph from the StoreMap schema, calculates shortest paths
via Dijkstra's algorithm, and calculates relative turn angles for blind navigation.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from navigation.map_schema import MapEdge, MapNode, StoreMap


@dataclass
class EdgeInfo:
    target_node_id: str
    distance_meters: float
    heading_deg: float
    notes: Optional[str] = None


class StoreGraph:
    """Navigable graph structure built from a StoreMap."""

    def __init__(self, store_map: StoreMap):
        self.store_map = store_map
        self.adjacency: Dict[str, List[EdgeInfo]] = {
            node_id: [] for node_id in store_map.nodes.keys()
        }
        self._build_graph()

    def _build_graph(self) -> None:
        """Populate adjacency list, automatically handling bidirectional edges."""
        for edge in self.store_map.edges:
            if edge.from_node not in self.adjacency:
                self.adjacency[edge.from_node] = []
            if edge.to_node not in self.adjacency:
                self.adjacency[edge.to_node] = []

            # Forward edge
            self.adjacency[edge.from_node].append(
                EdgeInfo(
                    target_node_id=edge.to_node,
                    distance_meters=edge.distance_meters,
                    heading_deg=edge.heading_deg,
                    notes=edge.notes,
                )
            )

            # Reverse edge if bidirectional
            if edge.bidirectional:
                reverse_heading = (edge.heading_deg + 180.0) % 360.0
                self.adjacency[edge.to_node].append(
                    EdgeInfo(
                        target_node_id=edge.from_node,
                        distance_meters=edge.distance_meters,
                        heading_deg=reverse_heading,
                        notes=edge.notes,
                    )
                )

    def find_shortest_path(self, start_node_id: str, goal_node_id: str) -> Optional[List[str]]:
        """Compute the shortest path using Dijkstra's algorithm.

        Returns:
            List of node IDs from start_node_id to goal_node_id, or None if no path exists.
        """
        if start_node_id not in self.store_map.nodes or goal_node_id not in self.store_map.nodes:
            return None

        if start_node_id == goal_node_id:
            return [start_node_id]

        distances: Dict[str, float] = {node_id: float("inf") for node_id in self.store_map.nodes}
        distances[start_node_id] = 0.0
        previous_nodes: Dict[str, Optional[str]] = {node_id: None for node_id in self.store_map.nodes}

        # Priority queue entries: (distance, node_id)
        pq: List[Tuple[float, str]] = [(0.0, start_node_id)]
        visited: Set[str] = set()

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue
            visited.add(current_node)

            if current_node == goal_node_id:
                break

            for edge in self.adjacency.get(current_node, []):
                neighbor = edge.target_node_id
                if neighbor in visited:
                    continue

                new_distance = current_distance + edge.distance_meters
                if new_distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(pq, (new_distance, neighbor))

        # Reconstruct path
        if distances.get(goal_node_id, float("inf")) == float("inf"):
            return None

        path: List[str] = []
        curr: Optional[str] = goal_node_id
        while curr is not None:
            path.append(curr)
            curr = previous_nodes.get(curr)

        path.reverse()
        return path

    def get_edge_between(self, from_node: str, to_node: str) -> Optional[EdgeInfo]:
        """Find the direct edge between two adjacent nodes."""
        for edge in self.adjacency.get(from_node, []):
            if edge.target_node_id == to_node:
                return edge
        return None

    def calculate_turn_angle(self, previous_heading_deg: float, next_heading_deg: float) -> float:
        """Calculate relative turn angle in degrees [-180, +180].

        Negative values = Turn Left
        Positive values = Turn Right
        """
        diff = (next_heading_deg - previous_heading_deg + 180.0) % 360.0 - 180.0
        return diff

    def classify_turn(self, turn_angle_deg: float) -> str:
        """Convert a relative turn angle in degrees to a natural spoken prompt."""
        abs_angle = abs(turn_angle_deg)
        if abs_angle <= 15.0:
            return "continue straight"
        elif turn_angle_deg > 0:
            if abs_angle <= 45.0:
                return "veer slightly right"
            elif abs_angle <= 115.0:
                return "turn right"
            elif abs_angle <= 155.0:
                return "turn sharp right"
            else:
                return "turn around (make a U-turn)"
        else:
            if abs_angle <= 45.0:
                return "veer slightly left"
            elif abs_angle <= 115.0:
                return "turn left"
            elif abs_angle <= 155.0:
                return "turn sharp left"
            else:
                return "turn around (make a U-turn)"

    def find_nearest_node(self, x: float, y: float) -> Optional[str]:
        """Find the closest node ID to Euclidean coordinates (x, y)."""
        best_node: Optional[str] = None
        best_dist = float("inf")
        for node_id, node in self.store_map.nodes.items():
            dist = math.hypot(node.x - x, node.y - y)
            if dist < best_dist:
                best_dist = dist
                best_node = node_id
        return best_node

    def validate_graph(self) -> List[str]:
        """Validate topological integrity. Returns list of warning/error messages."""
        issues: List[str] = []

        if not self.store_map.nodes:
            issues.append("Store map has no nodes defined.")
            return issues

        # Check for missing node references in edges
        for i, edge in enumerate(self.store_map.edges):
            if edge.from_node not in self.store_map.nodes:
                issues.append(f"Edge #{i} references non-existent from_node '{edge.from_node}'")
            if edge.to_node not in self.store_map.nodes:
                issues.append(f"Edge #{i} references non-existent to_node '{edge.to_node}'")

        # Check for unreachable nodes from entrance
        entrance_nodes = [nid for nid, n in self.store_map.nodes.items() if n.node_type.value == "entrance"]
        if not entrance_nodes:
            issues.append("Warning: No node with type 'entrance' found.")
        else:
            start_entrance = entrance_nodes[0]
            for nid in self.store_map.nodes:
                if nid != start_entrance and self.find_shortest_path(start_entrance, nid) is None:
                    issues.append(f"Node '{nid}' is disconnected/unreachable from entrance '{start_entrance}'.")

        return issues

