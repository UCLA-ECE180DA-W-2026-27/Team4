"""Pydantic data schemas for store navigation and mapping.

Defines the structure for nodes, edges, products, landmarks, and store metadata
used to guide visually impaired users through stores like CVS.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ShelfSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    ENDCAP = "endcap"
    ISLAND = "island"
    FRONT = "front"


class ShelfHeight(str, Enum):
    TOP = "top"
    EYE_LEVEL = "eye_level"
    MIDDLE = "middle"
    BOTTOM = "bottom"


class NodeType(str, Enum):
    ENTRANCE = "entrance"
    INTERSECTION = "intersection"
    AISLE_WAYPOINT = "aisle_waypoint"
    SHELF_TARGET = "shelf_target"
    CHECKOUT = "checkout"
    EXIT = "exit"
    INFO_DESK = "info_desk"


class VisualLandmark(BaseModel):
    """Visual anchor used for position confirmation and drift reset."""
    landmark_id: str
    landmark_type: str = "overhead_aisle_sign"  # overhead_aisle_sign, banner, counter, door
    text: Optional[str] = None  # e.g., "AISLE 4 SKIN CARE"
    expected_bearing_deg: Optional[float] = None  # Expected heading to look at the sign
    description: Optional[str] = None


class ProductPlacement(BaseModel):
    """A product or product group located at a specific shelf/waypoint."""
    product_id: str
    name: str  # e.g. "CeraVe Moisturizing Cream"
    category: str  # e.g. "moisturizer"
    keywords: List[str] = Field(default_factory=list)  # e.g. ["lotion", "cream", "dry skin", "cerave"]
    shelf_side: ShelfSide = ShelfSide.RIGHT
    shelf_height: ShelfHeight = ShelfHeight.MIDDLE
    brand: Optional[str] = None
    description: Optional[str] = None


class MapNode(BaseModel):
    """A waypoint or point of interest in the store."""
    node_id: str
    name: str  # e.g., "Aisle 4 - Skincare Section"
    node_type: NodeType = NodeType.AISLE_WAYPOINT
    x: float = Field(..., description="X coordinate in meters relative to store entrance")
    y: float = Field(..., description="Y coordinate in meters relative to store entrance")
    landmarks: List[VisualLandmark] = Field(default_factory=list)
    products: List[ProductPlacement] = Field(default_factory=list)
    description: Optional[str] = None


class MapEdge(BaseModel):
    """Navigable path segment connecting two nodes."""
    from_node: str
    to_node: str
    distance_meters: float = Field(..., gt=0)
    heading_deg: float = Field(..., ge=0.0, lt=360.0, description="Compass heading from from_node to to_node (0=North, 90=East)")
    bidirectional: bool = True
    approx_steps: Optional[int] = None
    notes: Optional[str] = None


class StoreMetadata(BaseModel):
    """Metadata describing the mapped retail location."""
    store_name: str = "CVS Pharmacy"
    store_id: str = "cvs_store"
    address: Optional[str] = None
    default_step_length_m: float = 0.70  # Average step length in meters
    north_reference_deg: float = 0.0  # Bearing alignment
    created_at: Optional[str] = None
    version: str = "1.0.0"


class StoreMap(BaseModel):
    """Complete graph and product database for a retail store."""
    metadata: StoreMetadata = Field(default_factory=StoreMetadata)
    nodes: Dict[str, MapNode] = Field(default_factory=dict)
    edges: List[MapEdge] = Field(default_factory=list)

    def get_node(self, node_id: str) -> Optional[MapNode]:
        return self.nodes.get(node_id)

    def calculate_steps(self, distance_m: float, step_length_m: Optional[float] = None) -> int:
        stride = step_length_m or self.metadata.default_step_length_m
        if stride <= 0:
            stride = 0.70
        return max(1, round(distance_m / stride))

