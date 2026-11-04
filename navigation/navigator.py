"""High-level navigation planner and turn-by-turn instruction generator.

Translates shortest-path waypoint routes into accessible, clear spoken prompts
tailored for visually impaired users navigating a retail store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from navigation.map_schema import MapNode, ProductPlacement, ShelfSide, StoreMap, VisualLandmark
from navigation.store_graph import StoreGraph


@dataclass
class NavigationStep:
    """A discrete leg or action in a route."""
    step_index: int
    action: str  # "orient", "walk", "turn_and_walk", "arrive", "face_shelf"
    verbal_instruction: str
    distance_meters: float = 0.0
    step_count: int = 0
    from_node_id: Optional[str] = None
    to_node_id: Optional[str] = None
    target_heading_deg: Optional[float] = None
    turn_angle_deg: Optional[float] = None
    landmarks_to_expect: List[VisualLandmark] = field(default_factory=list)
    shelf_details: Optional[str] = None


@dataclass
class NavigationPlan:
    """Full end-to-end route plan with spoken instructions and milestones."""
    destination_node_id: str
    destination_name: str
    target_product: Optional[ProductPlacement]
    total_distance_meters: float
    total_steps: int
    path_node_ids: List[str]
    steps: List[NavigationStep]

    def get_summary_prompt(self) -> str:
        """Overview spoken to the user right before departure."""
        prod_text = f" for {self.target_product.name}" if self.target_product else ""
        return (
            f"Route planned to {self.destination_name}{prod_text}. "
            f"Total distance is approximately {self.total_steps} steps, "
            f"about {round(self.total_distance_meters, 1)} meters."
        )


class StoreNavigator:
    """Generates accessible turn-by-turn guidance from a StoreGraph."""

    def __init__(self, store_map: StoreMap):
        self.store_map = store_map
        self.graph = StoreGraph(store_map)

    def plan_route(
        self,
        start_node_id: str,
        goal_node_id: str,
        initial_heading_deg: Optional[float] = None,
        target_product: Optional[ProductPlacement] = None,
        user_step_length_m: Optional[float] = None,
    ) -> Optional[NavigationPlan]:
        """Compute the path and synthesize natural spoken directions.

        Args:
            start_node_id: Starting waypoint ID (e.g. "entrance")
            goal_node_id: Target waypoint ID (e.g. "aisle_4_skincare")
            initial_heading_deg: User's initial heading in degrees (0-360), if known
            target_product: Optional product being navigated to
            user_step_length_m: Custom user step length in meters

        Returns:
            NavigationPlan with turn-by-turn verbal instructions, or None if no path.
        """
        path = self.graph.find_shortest_path(start_node_id, goal_node_id)
        if not path:
            return None

        dest_node = self.store_map.nodes[goal_node_id]
        stride = user_step_length_m or self.store_map.metadata.default_step_length_m

        steps: List[NavigationStep] = []
        total_dist = 0.0
        total_steps = 0
        current_heading = initial_heading_deg

        # Handle starting at destination
        if len(path) == 1:
            arrive_text = f"You are already at {dest_node.name}."
            if target_product:
                arrive_text += f" {target_product.name} is on your {target_product.shelf_side.value} side."
            steps.append(
                NavigationStep(
                    step_index=0,
                    action="arrive",
                    verbal_instruction=arrive_text,
                    from_node_id=start_node_id,
                    to_node_id=start_node_id,
                )
            )
            return NavigationPlan(
                destination_node_id=goal_node_id,
                destination_name=dest_node.name,
                target_product=target_product,
                total_distance_meters=0.0,
                total_steps=0,
                path_node_ids=path,
                steps=steps,
            )

        step_counter = 0

        # Traverse each leg of the path
        for i in range(len(path) - 1):
            curr_id = path[i]
            next_id = path[i + 1]
            curr_node = self.store_map.nodes[curr_id]
            next_node = self.store_map.nodes[next_id]

            edge = self.graph.get_edge_between(curr_id, next_id)
            if not edge:
                continue

            dist_m = edge.distance_meters
            steps_count = self.store_map.calculate_steps(dist_m, stride)
            total_dist += dist_m
            total_steps += steps_count

            leg_heading = edge.heading_deg

            # Determine turn required
            if current_heading is None:
                # First leg with unknown initial heading
                verbal = f"Walk forward {steps_count} steps towards {next_node.name}."
                turn_angle = None
            else:
                turn_angle = self.graph.calculate_turn_angle(current_heading, leg_heading)
                turn_desc = self.graph.classify_turn(turn_angle)

                if turn_desc == "continue straight":
                    verbal = f"Continue straight for {steps_count} steps towards {next_node.name}."
                else:
                    verbal = f"{turn_desc.capitalize()}, then walk forward {steps_count} steps towards {next_node.name}."

            # Append landmark cues if the next node has signs
            if next_node.landmarks:
                sign = next_node.landmarks[0]
                if sign.text:
                    verbal += f" You should see overhead sign: '{sign.text}'."

            steps.append(
                NavigationStep(
                    step_index=step_counter,
                    action="turn_and_walk" if (turn_angle and abs(turn_angle) > 15) else "walk",
                    verbal_instruction=verbal,
                    distance_meters=dist_m,
                    step_count=steps_count,
                    from_node_id=curr_id,
                    to_node_id=next_id,
                    target_heading_deg=leg_heading,
                    turn_angle_deg=turn_angle,
                    landmarks_to_expect=next_node.landmarks,
                )
            )
            step_counter += 1
            current_heading = leg_heading

        # Final arrival and shelf alignment instruction
        arrival_prompt = f"You have arrived at {dest_node.name}."
        shelf_details = None

        if target_product:
            side = target_product.shelf_side.value
            height = target_product.shelf_height.value.replace("_", " ")

            if side in ("left", "right"):
                face_dir = "left" if side == "left" else "right"
                arrival_prompt += (
                    f" {target_product.name} is on your {side}, at {height} height. "
                    f"Turn 90 degrees {face_dir} to face the shelf directly."
                )
            elif side == "endcap":
                arrival_prompt += f" {target_product.name} is on the endcap shelf in front of you."
            else:
                arrival_prompt += f" {target_product.name} is on the shelf at {height} height."

            arrival_prompt += " When ready, ask 'What is on the shelf?' to scan."
            shelf_details = f"Side: {side}, Height: {height}"

        steps.append(
            NavigationStep(
                step_index=step_counter,
                action="arrive",
                verbal_instruction=arrival_prompt,
                from_node_id=path[-1],
                to_node_id=path[-1],
                target_heading_deg=current_heading,
                shelf_details=shelf_details,
            )
        )

        return NavigationPlan(
            destination_node_id=goal_node_id,
            destination_name=dest_node.name,
            target_product=target_product,
            total_distance_meters=total_dist,
            total_steps=total_steps,
            path_node_ids=path,
            steps=steps,
        )

