"""Unit tests for dead reckoning and visual landmark drift correction."""

import unittest
from navigation.localization import LocalizationTracker, NavStatus
from navigation.map_serializer import MapSerializer
from navigation.navigator import StoreNavigator


class TestLocalization(unittest.TestCase):

    def setUp(self):
        self.store_map = MapSerializer.load_from_json("data/sample_cvs_map.json")
        self.navigator = StoreNavigator(self.store_map)
        self.tracker = LocalizationTracker(self.store_map)

    def test_dead_reckoning_progression(self):
        plan = self.navigator.plan_route("entrance", "aisle_4_skincare", initial_heading_deg=0.0)
        self.assertIsNotNone(plan)

        state = self.tracker.start_navigation(plan, initial_heading_deg=0.0)
        self.assertEqual(state.status, NavStatus.NAVIGATING)
        self.assertEqual(state.current_node_id, "entrance")

        # Take steps along the first leg (entrance -> front_lobby, 5 steps)
        for _ in range(4):
            state, prompt = self.tracker.on_step_detected(step_heading_deg=0.0)
            self.assertEqual(state.status, NavStatus.NAVIGATING)
            self.assertIsNone(prompt)

        # 5th step should complete leg 1 and prompt next direction
        state, prompt = self.tracker.on_step_detected(step_heading_deg=0.0)
        self.assertIsNotNone(prompt)
        self.assertTrue("Continue straight" in prompt or "Main Center Hallway" in prompt)

    def test_visual_landmark_drift_correction(self):
        plan = self.navigator.plan_route("entrance", "aisle_4_skincare", initial_heading_deg=0.0)
        self.tracker.start_navigation(plan, initial_heading_deg=0.0)

        # Simulate camera OCR detecting overhead sign: "AISLE 4 SKIN CARE"
        corrected, prompt = self.tracker.on_visual_landmark_detected("AISLE 4 SKIN CARE")
        self.assertTrue(corrected)
        self.assertEqual(self.tracker.state.current_node_id, "aisle_4_entrance")
        self.assertEqual(self.tracker.state.drift_corrections_applied, 1)
        self.assertIn("Aisle 4", prompt)


if __name__ == "__main__":
    unittest.main()
