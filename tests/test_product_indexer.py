"""Unit tests for product indexing and fuzzy search."""

import unittest
from navigation.map_serializer import MapSerializer
from navigation.product_indexer import ProductIndexer


class TestProductIndexer(unittest.TestCase):

    def setUp(self):
        self.store_map = MapSerializer.load_from_json("data/sample_cvs_map.json")
        self.indexer = ProductIndexer(self.store_map)

    def test_exact_and_fuzzy_product_search(self):
        # Direct name search
        results = self.indexer.search("CeraVe Moisturizing Cream")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].product.name, "CeraVe Moisturizing Cream")
        self.assertEqual(results[0].node_id, "aisle_4_skincare")

        # Category search
        results_moisturizer = self.indexer.search("moisturizer")
        self.assertGreaterEqual(len(results_moisturizer), 3)
        found_names = [r.product.name for r in results_moisturizer]
        self.assertIn("CeraVe Moisturizing Cream", found_names)
        self.assertIn("Neutrogena Hydro Boost Water Gel", found_names)

        # Natural language spoken query with conversational filler
        results_nl = self.indexer.search("Where is the sunscreen?")
        self.assertGreater(len(results_nl), 0)
        self.assertTrue(
            "sunscreen" in results_nl[0].product.name.lower() or results_nl[0].product.category == "sunscreen"
        )

        # Synonyms / Brand fuzzy search
        results_advil = self.indexer.search("i need ibuprofen")
        self.assertGreater(len(results_advil), 0)
        self.assertTrue(
            "advil" in results_advil[0].product.name.lower() or "ibuprofen" in results_advil[0].product.name.lower()
        )


if __name__ == "__main__":
    unittest.main()
