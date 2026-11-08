"""Real-time localization and state tracker for retail navigation.

Fuses Pedestrian Dead Reckoning (IMU step events + compass heading) with visual
landmark confirmations (overhead aisle signs via OCR) to track position and eliminate drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from rapidfuzz import fuzz
from navigation.map_schema import MapNode, StoreMap, VisualLandmark
from navigation.navigator import NavigationPlan, NavigationStep


class NavStatus(str, Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    LEG_COMPLETED = "leg_completed"
    ARRIVED = "arrived"
    OFF_ROUTE = "off_route"


@dataclass
class LocalizationState:
    """Current live estimation of the user's spatial location in the store."""
    current_node_id: str
    target_node_id: Optional[str]
    current_leg_index: int
    steps_taken_on_leg: int
    steps_remaining_on_leg: int
    estimated_x: float
    estimated_y: float
    current_heading_deg: float
    status: NavStatus
    last_spoken_prompt: Optional[str] = None
    drift_corrections_applied: int = 0


class LocalizationTracker:
    """Tracks user progress through a NavigationPlan using step events and visual landmarks."""

    def __init__(self, store_map: StoreMap):
        self.store_map = store_map
        self.plan: Optional[NavigationPlan] = None
        self.state: Optional[LocalizationState] = None

    def start_navigation(self, plan: NavigationPlan, initial_heading_deg: float = 0.0) -> LocalizationState:
        """Initialize tracker with a newly generated NavigationPlan."""
        self.plan = plan
        first_node_id = plan.path_node_ids[0]
        first_node = self.store_map.nodes[first_node_id]

        first_step = plan.steps[0] if plan.steps else None
        target_node = first_step.to_node_id if first_step else first_node_id
        steps_total = first_step.step_count if first_step else 0

        self.state = LocalizationState(
            current_node_id=first_node_id,
            target_node_id=target_node,
            current_leg_index=0,
            steps_taken_on_leg=0,
            steps_remaining_on_leg=steps_total,
            estimated_x=first_node.x,
            estimated_y=first_node.y,
            current_heading_deg=initial_heading_deg,
            status=NavStatus.NAVIGATING,
            last_spoken_prompt=first_step.verbal_instruction if first_step else None,
            drift_corrections_applied=0,
        )
        return self.state

    def on_step_detected(self, step_heading_deg: float) -> Tuple[LocalizationState, Optional[str]]:
        """Called whenever the IMU detects a completed pedestrian step.

        Args:
            step_heading_deg: Current compass heading of the step from BerryIMU.

        Returns:
            Tuple of (updated LocalizationState, optional spoken prompt if milestone reached).
        """
        if not self.plan or not self.state or self.state.status != NavStatus.NAVIGATING:
            return self.state, None

        self.state.current_heading_deg = step_heading_deg
        self.state.steps_taken_on_leg += 1
        self.state.steps_remaining_on_leg = max(0, self.state.steps_remaining_on_leg - 1)

        # Update continuous coordinates (dead reckoning)
        step_len = self.store_map.metadata.default_step_length_m
        rad = math.radians(step_heading_deg)
        # 0 deg = North (+Y), 90 deg = East (+X)
        self.state.estimated_x += step_len * math.sin(rad)
        self.state.estimated_y += step_len * math.cos(rad)

        spoken_prompt = None

        # Check if current leg is completed
        if self.state.steps_remaining_on_leg == 0:
            next_leg_idx = self.state.current_leg_index + 1

            if next_leg_idx < len(self.plan.steps):
                next_step = self.plan.steps[next_leg_idx]
                self.state.current_leg_index = next_leg_idx
                self.state.current_node_id = next_step.from_node_id or self.state.current_node_id
                self.state.target_node_id = next_step.to_node_id
                self.state.steps_taken_on_leg = 0
                self.state.steps_remaining_on_leg = next_step.step_count

                if next_step.action == "arrive":
                    self.state.status = NavStatus.ARRIVED
                    spoken_prompt = next_step.verbal_instruction
                else:
                    self.state.status = NavStatus.NAVIGATING
                    spoken_prompt = next_step.verbal_instruction

                self.state.last_spoken_prompt = spoken_prompt
            else:
                self.state.status = NavStatus.ARRIVED

        return self.state, spoken_prompt

    def on_visual_landmark_detected(self, detected_text: str, similarity_threshold: float = 75.0) -> Tuple[bool, Optional[str]]:
        """Called when overhead camera / OCR detects an aisle sign or landmark text.

        Fuses visual detection with topological map to correct any dead-reckoning drift.

        Returns:
            Tuple of (was_corrected: bool, optional prompt explaining correction).
        """
        if not self.plan or not self.state:
            return False, None

        clean_detected = detected_text.strip().lower()
        best_match_node: Optional[MapNode] = None
        best_score = 0.0

        # Check landmarks across nodes
        for node_id, node in self.store_map.nodes.items():
            for lm in node.landmarks:
                if lm.text:
                    score = fuzz.token_set_ratio(clean_detected, lm.text.lower())
                    if score > best_score:
                        best_score = score
                        best_match_node = node

        if best_match_node and best_score >= similarity_threshold:
            # Snap position to the visual landmark node
            old_x, old_y = self.state.estimated_x, self.state.estimated_y
            self.state.estimated_x = best_match_node.x
            self.state.estimated_y = best_match_node.y
            self.state.current_node_id = best_match_node.node_id
            self.state.drift_corrections_applied += 1

            # Check if this landmark corresponds to the target node of the current leg
            if self.state.target_node_id == best_match_node.node_id:
                # User arrived early or on time visually!
                self.state.steps_remaining_on_leg = 0
                return True, f"Visual confirmation: Arrived at {best_match_node.name}."
            else:
                return True, f"Visual confirmation: Position verified at {best_match_node.name}."

        return False, None

