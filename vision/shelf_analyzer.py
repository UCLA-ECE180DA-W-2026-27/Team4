"""Shelf analysis session manager coordinating Gemini VLM, Q&A dialogue, and target selection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import cv2

from vision.gemini_client import DetectedProduct, GeminiShelfClient, ShelfInspectionResult
from vision.visualizer import ShelfVisualizer


class ShelfAnalyzer:
    """Coordinates shelf image inspection, multi-turn dialogue, and target locking."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = GeminiShelfClient(api_key=api_key)
        self.current_image_path: Optional[Path] = None
        self.current_inspection: Optional[ShelfInspectionResult] = None
        self.selected_target: Optional[DetectedProduct] = None

    def inspect_shelf(self, image_path: str | Path) -> ShelfInspectionResult:
        """Inspect a shelf image and update session state."""
        path = Path(image_path)
        self.current_image_path = path
        self.current_inspection = self.client.inspect_shelf_image(path)
        self.selected_target = None
        return self.current_inspection

    def ask(self, question: str) -> str:
        """Ask a conversational question about the visible shelf products."""
        if not self.current_inspection:
            return "Please capture or inspect a shelf image first."

        # Check if user is expressing a choice to pick an item
        target = self.find_product(question)
        if target and any(w in question.lower() for w in ["pick", "grab", "select", "want", "guide", "choose"]):
            self.selected_target = target
            return (
                f"Target locked on {target.name}. "
                f"It is on the {target.side} side of the {target.shelf_level} shelf. "
                f"Reach your hand into view to begin guidance."
            )

        return self.client.ask_question(question)

    def select_target(self, product_id_or_name: str) -> Optional[DetectedProduct]:
        """Explicitly set the active target product for visual servoing."""
        target = self.find_product(product_id_or_name)
        if target:
            self.selected_target = target
        return target

    def find_product(self, query: str) -> Optional[DetectedProduct]:
        """Find a product in the current inspection matching query or ID."""
        if not self.current_inspection:
            return None

        # Check by product ID
        p_by_id = self.current_inspection.get_product_by_id(query)
        if p_by_id:
            return p_by_id

        # Check by name/brand
        return self.current_inspection.find_product_by_name(query)

    def get_target_pixel_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Get (x1, y1, x2, y2) pixel bounding box of currently selected target."""
        if not self.selected_target or not self.current_image_path:
            return None

        img = cv2.imread(str(self.current_image_path))
        if img is None:
            return None

        h, w = img.shape[:2]
        ymin, xmin, ymax, xmax = self.selected_target.box_2d
        x1 = int(xmin * w / 1000.0)
        y1 = int(ymin * h / 1000.0)
        x2 = int(xmax * w / 1000.0)
        y2 = int(ymax * h / 1000.0)
        return (x1, y1, x2, y2)

    def render_annotated_image(self, output_path: str | Path) -> None:
        """Render and save annotated shelf image with bounding boxes."""
        if not self.current_image_path or not self.current_inspection:
            raise ValueError("No inspection available to render.")

        target_id = self.selected_target.product_id if self.selected_target else None
        ShelfVisualizer.draw_detections(
            image_path=self.current_image_path,
            inspection=self.current_inspection,
            target_product_id=target_id,
            output_path=output_path,
        )

