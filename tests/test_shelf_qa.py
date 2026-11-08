"""Unit tests for Gemini shelf inspection, Q&A dialogue, and visualizer."""

import unittest
from pathlib import Path

from vision import DetectedProduct, GeminiShelfClient, ShelfAnalyzer, ShelfVisualizer


class TestShelfQA(unittest.TestCase):

    def setUp(self):
        self.image_path = Path("data/sample_cvs_shelf.jpg")
        self.analyzer = ShelfAnalyzer()

    def test_shelf_inspection_products_and_summary(self):
        inspection = self.analyzer.inspect_shelf(self.image_path)

        self.assertIsNotNone(inspection)
        self.assertGreater(len(inspection.spoken_summary), 20)
        self.assertGreaterEqual(len(inspection.products), 3)

        # Check product data completeness
        for p in inspection.products:
            self.assertTrue(len(p.name) > 0)
            self.assertTrue(len(p.brand) > 0)
            self.assertEqual(len(p.box_2d), 4)
            ymin, xmin, ymax, xmax = p.box_2d
            self.assertGreaterEqual(ymin, 0)
            self.assertLessEqual(ymax, 1000)
            self.assertGreater(ymax, ymin)
            self.assertGreater(xmax, xmin)

    def test_conversational_qa(self):
        self.analyzer.inspect_shelf(self.image_path)

        # Query oil-free
        ans_oil = self.analyzer.ask("Which one is oil-free?")
        self.assertIn("Neutrogena", ans_oil)

        # Query dry skin
        ans_dry = self.analyzer.ask("What do you recommend for dry skin?")
        self.assertIn("CeraVe", ans_dry)

    def test_target_locking_and_coordinates(self):
        self.analyzer.inspect_shelf(self.image_path)

        # Target selection query
        ans_target = self.analyzer.ask("I want the CeraVe, please guide my hand")
        self.assertIn("Target locked", ans_target)
        self.assertIsNotNone(self.analyzer.selected_target)
        self.assertEqual(self.analyzer.selected_target.brand, "CeraVe")

        # Pixel bounding box calculation
        pixel_box = self.analyzer.get_target_pixel_box()
        self.assertIsNotNone(pixel_box)
        x1, y1, x2, y2 = pixel_box
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)

    def test_shelf_visualizer_rendering(self):
        inspection = self.analyzer.inspect_shelf(self.image_path)
        test_out = Path("data/test_detection_output.jpg")

        ShelfVisualizer.draw_detections(
            image_path=self.image_path,
            inspection=inspection,
            target_product_id="p1",
            output_path=test_out,
        )

        self.assertTrue(test_out.exists())
        self.assertGreater(test_out.stat().st_size, 1000)

        # Clean up
        test_out.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

