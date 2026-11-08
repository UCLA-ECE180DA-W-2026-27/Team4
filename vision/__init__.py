"""Vision module for shelf inspection, Gemini VLM analysis, and visual guidance."""

from vision.gemini_client import (
    DetectedProduct,
    GeminiShelfClient,
    ShelfInspectionResult,
)
from vision.shelf_analyzer import ShelfAnalyzer
from vision.visualizer import ShelfVisualizer

__all__ = [
    "DetectedProduct",
    "GeminiShelfClient",
    "ShelfInspectionResult",
    "ShelfAnalyzer",
    "ShelfVisualizer",
]

