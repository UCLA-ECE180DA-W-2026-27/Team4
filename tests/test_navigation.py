"""Unit tests for Dijkstra shortest path and turn-by-turn guidance."""

import unittest
from navigation.map_serializer import MapSerializer
from navigation.navigator import StoreNavigator
from navigation.store_graph import StoreGraph


class TestNavigation(unittest.TestCase):

    def setUp(self):
        self.store_map = MapSerializer.load_from_json("data/sample_cvs_map.json")
        self.graph = StoreGraph(self.store_map)
        self.navigator = StoreNavigator(self.store_map)

    def test_dijkstra_shortest_path(self):
        # Route from entrance to moisturizer
        path = self.graph.find_shortest_path("entrance", "aisle_4_skincare")
        self.assertEqual(path, ["entrance", "front_lobby", "main_hallway_center", "aisle_4_entrance", "aisle_4_skincare"])

        # Route to pharmacy
        path_pharmacy = self.graph.find_shortest_path("entrance", "pharmacy_counter")
        self.assertIn("pharmacy_counter", path_pharmacy)
        self.assertEqual(path_pharmacy[0], "entrance")

    def test_turn_angle_classification(self):
        # 0 deg heading turning to 90 deg -> turn right (+90)
        angle_right = self.graph.calculate_turn_angle(0.0, 90.0)
        self.assertEqual(angle_right, 90.0)
        self.assertEqual(self.graph.classify_turn(angle_right), "turn right")

        # 0 deg heading turning to 270 deg -> turn left (-90)
        angle_left = self.graph.calculate_turn_angle(0.0, 270.0)
        self.assertEqual(angle_left, -90.0)
        self.assertEqual(self.graph.classify_turn(angle_left), "turn left")

        # Straight
        angle_straight = self.graph.calculate_turn_angle(0.0, 5.0)
        self.assertEqual(self.graph.classify_turn(angle_straight), "continue straight")

    def test_turn_by_turn_plan_generation(self):
        plan = self.navigator.plan_route(
            start_node_id="entrance",
            goal_node_id="aisle_4_skincare",
            initial_heading_deg=0.0,
        )

        self.assertIsNotNone(plan)
        self.assertGreater(plan.total_distance_meters, 15.0)
        self.assertGreater(plan.total_steps, 20)
        self.assertGreaterEqual(len(plan.steps), 5)

        # Check that verbal instructions exist for all legs
        for step in plan.steps:
            self.assertGreater(len(step.verbal_instruction), 10)
            self.assertTrue(
                "steps" in step.verbal_instruction.lower() or "arrived" in step.verbal_instruction.lower()
            )


if __name__ == "__main__":
    unittest.main()
