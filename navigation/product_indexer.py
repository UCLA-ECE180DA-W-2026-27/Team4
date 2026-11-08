"""Product indexer and fuzzy search engine for retail store maps.

Enables natural language voice queries like "Where is the moisturizer?" or
"Find CeraVe cream" to be resolved to specific store nodes, aisles, and shelf coordinates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from rapidfuzz import fuzz

from navigation.map_schema import MapNode, ProductPlacement, ShelfHeight, ShelfSide, StoreMap


@dataclass
class ProductSearchResult:
    product: ProductPlacement
    node_id: str
    node_name: str
    shelf_side: ShelfSide
    shelf_height: ShelfHeight
    match_score: float
    matched_term: str


# Common retail product synonym mappings to improve voice matching
COMMON_SYNONYMS: Dict[str, List[str]] = {
    "moisturizer": ["lotion", "face cream", "moisturizing cream", "hydrating lotion", "body lotion"],
    "sunscreen": ["sunblock", "spf", "sun cream", "sun protection"],
    "pain reliever": ["headache pills", "pain medicine", "fever reducer"],
    "cold medicine": ["cough syrup", "decongestant", "cold and flu"],
    "shampoo": ["hair wash", "conditioner", "hair care"],
    "soap": ["body wash", "bar soap", "shower gel", "hand soap"],
    "toothpaste": ["dental cream", "oral care", "toothbrush", "floss"],
    "bandaids": ["bandages", "first aid", "gauze", "wound care"],
}


class ProductIndexer:
    """Indexes products across all store map nodes and resolves natural language queries."""

    def __init__(self, store_map: StoreMap):
        self.store_map = store_map
        self._index: List[Dict] = []
        self._build_index()

    def _build_index(self) -> None:
        """Extract all products with their location, keywords, and synonyms."""
        self._index.clear()
        for node_id, node in self.store_map.nodes.items():
            for product in node.products:
                # Direct terms (highest specificity)
                direct_terms: Set[str] = {
                    product.name.lower(),
                }
                if product.brand:
                    direct_terms.add(product.brand.lower())
                for kw in product.keywords:
                    direct_terms.add(kw.lower())

                # Secondary terms (category & synonyms)
                secondary_terms: Set[str] = {
                    product.category.lower(),
                }
                for canonical, synonyms in COMMON_SYNONYMS.items():
                    if canonical in product.category.lower() or canonical in product.name.lower():
                        secondary_terms.update(s.lower() for s in synonyms)

                self._index.append({
                    "product": product,
                    "node_id": node_id,
                    "node_name": node.name,
                    "direct_terms": list(direct_terms),
                    "secondary_terms": list(secondary_terms),
                    "full_text": f"{product.name} {product.category} {product.brand or ''} {' '.join(product.keywords)}".lower(),
                })

    def search(self, query: str, score_cutoff: float = 60.0, limit: int = 5) -> List[ProductSearchResult]:
        """Search products using fuzzy string matching.

        Args:
            query: The spoken user query (e.g. "moisturizer", "CeraVe cream")
            score_cutoff: Minimum similarity threshold (0-100)
            limit: Maximum number of results to return

        Returns:
            Ranked list of ProductSearchResult objects.
        """
        clean_query = self._clean_query(query)
        if not clean_query:
            return []

        results: List[ProductSearchResult] = []

        for entry in self._index:
            product: ProductPlacement = entry["product"]
            direct_terms: List[str] = entry["direct_terms"]
            secondary_terms: List[str] = entry["secondary_terms"]

            best_score = 0.0
            best_matched_term = ""

            # Check direct terms (name, brand, keywords) -> receives high priority
            for term in direct_terms:
                s1 = fuzz.token_set_ratio(clean_query, term)
                s2 = fuzz.partial_ratio(clean_query, term)
                eff = max(s1, s2 * 0.95)
                # Boost direct term matches slightly so specific brand/names outrank generic categories
                boosted = min(100.0, eff * 1.05) if eff > 75.0 else eff
                if boosted > best_score:
                    best_score = boosted
                    best_matched_term = term

            # Check full text
            full_score = fuzz.token_set_ratio(clean_query, entry["full_text"])
            if full_score > best_score:
                best_score = full_score
                best_matched_term = product.name

            # Check secondary terms (category, broad synonyms)
            for term in secondary_terms:
                s1 = fuzz.token_set_ratio(clean_query, term)
                s2 = fuzz.partial_ratio(clean_query, term)
                eff = max(s1, s2 * 0.9)
                # Secondary matches have max base score 95 to allow direct matches to win ties
                eff = min(95.0, eff)
                if eff > best_score:
                    best_score = eff
                    best_matched_term = term

            if best_score >= score_cutoff:
                results.append(
                    ProductSearchResult(
                        product=product,
                        node_id=entry["node_id"],
                        node_name=entry["node_name"],
                        shelf_side=product.shelf_side,
                        shelf_height=product.shelf_height,
                        match_score=best_score,
                        matched_term=best_matched_term,
                    )
                )

        # Sort descending by match score
        results.sort(key=lambda r: r.match_score, reverse=True)
        return results[:limit]

    def _clean_query(self, query: str) -> str:
        """Strip conversational filler words from spoken query."""
        clean = query.lower().strip()
        prefixes = [
            "where is the", "where is", "where are the", "where are",
            "how do i get to the", "how do i get to",
            "take me to the", "take me to",
            "find me some", "find some", "find the", "find",
            "look for", "looking for", "i need", "i want", "can you find",
        ]
        for p in prefixes:
            if clean.startswith(p):
                clean = clean[len(p):].strip()
                break

        clean = re.sub(r"[^\w\s]", " ", clean).strip()
        return clean
