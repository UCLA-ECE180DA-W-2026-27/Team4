"""Unit tests for store map schema and validation."""

import unittest
from navigation.map_schema import (
    MapEdge,
    MapNode,
    NodeType,
    ProductPlacement,
    ShelfHeight,
    ShelfSide,
    StoreMap,
    StoreMetadata,
)
from navigation.store_graph import StoreGraph


class TestMapSchema(unittest.TestCase):

    def test_valid_store_map_creation(self):
        store_map = StoreMap(
            metadata=StoreMetadata(store_name="Test Store", default_step_length_m=0.7),
            nodes={
                "entrance": MapNode(node_id="entrance", name="Entrance", node_type=NodeType.ENTRANCE, x=0.0, y=0.0),
                "aisle_1": MapNode(
                    node_id="aisle_1",
                    name="Aisle 1",
                    node_type=NodeType.SHELF_TARGET,
                    x=0.0,
                    y=7.0,
                    products=[
                        ProductPlacement(
                            product_id="p1",
                            name="CeraVe Lotion",
                            category="moisturizer",
                            shelf_side=ShelfSide.RIGHT,
                            shelf_height=ShelfHeight.MIDDLE,
                        )
                    ],
                ),
            },
            edges=[
                MapEdge(from_node="entrance", to_node="aisle_1", distance_meters=7.0, heading_deg=0.0, bidirectional=True)
            ],
        )

        self.assertEqual(len(store_map.nodes), 2)
        self.assertEqual(len(store_map.edges), 1)
        self.assertEqual(store_map.calculate_steps(7.0), 10)  # 7.0 / 0.7 = 10 steps

        graph = StoreGraph(store_map)
        issues = graph.validate_graph()
        self.assertEqual(len(issues), 0)

    def test_invalid_edge_references(self):
        store_map = StoreMap(
            nodes={
                "entrance": MapNode(node_id="entrance", name="Entrance", node_type=NodeType.ENTRANCE, x=0.0, y=0.0)
            },
            edges=[
                MapEdge(from_node="entrance", to_node="ghost_node", distance_meters=5.0, heading_deg=0.0)
            ],
        )
        graph = StoreGraph(store_map)
        issues = graph.validate_graph()
        self.assertTrue(any("ghost_node" in issue for issue in issues))

    def test_disconnected_node_detection(self):
        store_map = StoreMap(
            nodes={
                "entrance": MapNode(node_id="entrance", name="Entrance", node_type=NodeType.ENTRANCE, x=0.0, y=0.0),
                "island_node": MapNode(node_id="island_node", name="Island", node_type=NodeType.AISLE_WAYPOINT, x=10.0, y=10.0),
            },
            edges=[],
        )
        graph = StoreGraph(store_map)
        issues = graph.validate_graph()
        self.assertTrue(any("disconnected/unreachable" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
