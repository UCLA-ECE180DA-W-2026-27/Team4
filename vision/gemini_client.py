"""Gemini Multimodal VLM Client for Retail Shelf Inspection & Conversational Q&A.

Communicates with Google Gemini Flash models to extract product listings,
2D bounding boxes, accessibility attributes, and maintain multi-turn shelf dialogue.
Supports live API calls via `google-genai` and realistic offline mock simulation.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
    from google.genai import types
    HAS_GENAI_SDK = True
except ImportError:
    HAS_GENAI_SDK = False


@dataclass
class DetectedProduct:
    product_id: str
    name: str
    brand: str
    category: str
    box_2d: List[int]  # [ymin, xmin, ymax, xmax] in normalized 0-1000 scale
    shelf_level: str  # top, eye_level, middle, bottom
    side: str  # left, center, right
    key_features: List[str] = field(default_factory=list)
    suitable_for: Optional[str] = None
    container_description: Optional[str] = None


@dataclass
class ShelfInspectionResult:
    spoken_summary: str
    products: List[DetectedProduct]
    raw_response: Optional[str] = None
    is_simulated: bool = False

    def get_product_by_id(self, pid: str) -> Optional[DetectedProduct]:
        for p in self.products:
            if p.product_id.lower() == pid.lower():
                return p
        return None

    def find_product_by_name(self, name_query: str) -> Optional[DetectedProduct]:
        query = name_query.lower().strip()
        for p in self.products:
            brand_l = p.brand.lower()
            name_l = p.name.lower()
            # Match if brand/name is inside query, or query is inside brand/name
            if brand_l in query or name_l in query or query in name_l or query in brand_l:
                return p
        return None


SHELF_ANALYSIS_SYSTEM_PROMPT = """You are an expert AI vision assistant helping a visually impaired shopper standing in front of a store shelf (e.g. CVS).
Analyze the captured shelf photograph and return a valid JSON object matching this schema:
{
  "spoken_summary": "Concise 2-3 sentence overview describing what products are visible, their shelf locations (left/center/right, top/middle/bottom), and prominent brands.",
  "products": [
    {
      "product_id": "p1",
      "name": "Full product name (e.g. CeraVe Moisturizing Cream)",
      "brand": "Brand name (e.g. CeraVe)",
      "category": "e.g. moisturizer / sunscreen / body wash",
      "box_2d": [ymin, xmin, ymax, xmax], // Bounding box normalized to 0-1000 integer range
      "shelf_level": "top | eye_level | middle | bottom",
      "side": "left | center | right",
      "key_features": ["fragrance-free", "with ceramides", "oil-free"],
      "suitable_for": "e.g. normal to dry skin",
      "container_description": "e.g. 16oz white tub with dark blue lid"
    }
  ]
}

Important Instructions:
- Keep the spoken_summary natural, clear, and direct so it can be spoken out loud immediately via text-to-speech.
- Include bounding boxes in [ymin, xmin, ymax, xmax] format (0 to 1000 scale).
- Output ONLY valid JSON, with no markdown code blocks or additional conversational text.
"""


class GeminiShelfClient:
    """Manages Gemini Multimodal VLM sessions for retail shelf scanning and Q&A."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        fallback_model: str = "gemini-1.5-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.client = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.last_inspection: Optional[ShelfInspectionResult] = None

        if self.api_key and HAS_GENAI_SDK:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[!] Warning: Could not initialize Gemini SDK client: {e}")

    def inspect_shelf_image(self, image_path: str | Path) -> ShelfInspectionResult:
        """Inspect a shelf photograph and extract structured product details with bounding boxes."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {path}")

        # Check if live API is available
        if self.client and self.api_key:
            try:
                return self._call_live_gemini(path)
            except Exception as e:
                print(f"[!] Live Gemini call failed ({e}). Falling back to simulation mode.")

        # Offline Mock / Simulation Mode
        return self._generate_simulated_shelf_result(path)

    def ask_question(self, user_question: str) -> str:
        """Ask a conversational follow-up question about the inspected shelf."""
        clean_q = user_question.strip()
        if not clean_q:
            return "Please ask a question about the items on the shelf."

        # If live API is configured, invoke chat with context
        if self.client and self.api_key and self.last_inspection and not self.last_inspection.is_simulated:
            try:
                prompt = (
                    f"The user asks: '{clean_q}'.\n"
                    f"Respond concisely in 1-2 friendly sentences suitable for text-to-speech, "
                    f"answering their question based on the visible shelf products."
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt],
                )
                return response.text.strip()
            except Exception as e:
                print(f"[!] Live Q&A call failed: {e}")

        # Simulated intelligent Q&A based on detected products
        return self._simulate_conversational_answer(clean_q)

    def _call_live_gemini(self, image_path: Path) -> ShelfInspectionResult:
        """Execute multimodal prompt on Gemini Flash using google-genai SDK."""
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        mime_type = "image/jpeg"
        if image_path.suffix.lower() == ".png":
            mime_type = "image/png"

        # Construct image part
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        prompt = "Analyze this retail shelf photo and output the required JSON catalog."

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                system_instruction=SHELF_ANALYSIS_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        raw_text = response.text.strip()
        result = self._parse_json_response(raw_text)
        self.last_inspection = result
        return result

    def _parse_json_response(self, text: str) -> ShelfInspectionResult:
        """Safely parse JSON response from Gemini."""
        clean = text.strip()
        # Strip ```json fences if present
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)

        data = json.loads(clean)
        summary = data.get("spoken_summary", "Here are the products found on this shelf.")
        products = []

        for p_data in data.get("products", []):
            products.append(
                DetectedProduct(
                    product_id=p_data.get("product_id", f"p_{len(products)+1}"),
                    name=p_data.get("name", "Unknown Product"),
                    brand=p_data.get("brand", "Unknown Brand"),
                    category=p_data.get("category", "skincare"),
                    box_2d=p_data.get("box_2d", [0, 0, 1000, 1000]),
                    shelf_level=p_data.get("shelf_level", "middle"),
                    side=p_data.get("side", "center"),
                    key_features=p_data.get("key_features", []),
                    suitable_for=p_data.get("suitable_for"),
                    container_description=p_data.get("container_description"),
                )
            )

        return ShelfInspectionResult(
            spoken_summary=summary,
            products=products,
            raw_response=text,
            is_simulated=False,
        )

    def _generate_simulated_shelf_result(self, image_path: Path) -> ShelfInspectionResult:
        """Return high-fidelity realistic CVS skincare shelf data when offline or in test mode."""
        simulated_products = [
            DetectedProduct(
                product_id="p1",
                name="CeraVe Moisturizing Cream",
                brand="CeraVe",
                category="moisturizer",
                box_2d=[380, 120, 720, 380],  # ymin, xmin, ymax, xmax
                shelf_level="middle",
                side="left",
                key_features=["ceramides", "hyaluronic acid", "fragrance-free", "barrier repair"],
                suitable_for="normal to very dry skin",
                container_description="16 oz wide white circular tub with dark blue lettering",
            ),
            DetectedProduct(
                product_id="p2",
                name="Neutrogena Hydro Boost Water Gel",
                brand="Neutrogena",
                category="moisturizer",
                box_2d=[340, 420, 680, 640],
                shelf_level="eye_level",
                side="center",
                key_features=["hyaluronic acid", "oil-free", "non-comedogenic", "water-light gel"],
                suitable_for="normal to oily or combination skin",
                container_description="1.7 oz vibrant blue circular jar",
            ),
            DetectedProduct(
                product_id="p3",
                name="Cetaphil Daily Hydrating Lotion",
                brand="Cetaphil",
                category="moisturizer",
                box_2d=[360, 680, 740, 890],
                shelf_level="middle",
                side="right",
                key_features=["hyaluronic acid", "hypoallergenic", "fragrance-free", "dermatologist tested"],
                suitable_for="sensitive or combination skin",
                container_description="3 fl oz slim white pump bottle with green accent",
            ),
        ]

        summary = (
            "In front of you on this shelf are three moisturizers: "
            "On the left, CeraVe Moisturizing Cream in a white and blue tub. "
            "In the center, Neutrogena Hydro Boost Water Gel in a blue jar. "
            "On the right, Cetaphil Daily Hydrating Lotion with a green cap."
        )

        result = ShelfInspectionResult(
            spoken_summary=summary,
            products=simulated_products,
            raw_response=None,
            is_simulated=True,
        )
        self.last_inspection = result
        return result

    def _simulate_conversational_answer(self, query: str) -> str:
        """Generate conversational answers based on the detected products."""
        q = query.lower()

        if "cerave" in q:
            return (
                "The CeraVe Moisturizing Cream is on the left side of the middle shelf in a 16 ounce white tub. "
                "It is fragrance-free, rich in ceramides, and ideal for dry skin."
            )
        elif "neutrogena" in q or "hydro boost" in q:
            return (
                "The Neutrogena Hydro Boost Water Gel is in the center at eye level in a blue jar. "
                "It is oil-free and absorbs quickly, making it great for oily or normal skin."
            )
        elif "cetaphil" in q:
            return (
                "The Cetaphil Daily Hydrating Lotion is on your right in a slim pump bottle. "
                "It is hypoallergenic and specially formulated for sensitive skin."
            )
        elif "oil free" in q or "oil-free" in q or "acne" in q or "oily" in q:
            return (
                "The Neutrogena Hydro Boost Water Gel is completely oil-free and non-comedogenic, "
                "so it won't clog pores. It is located in the center of the shelf."
            )
        elif "fragrance free" in q or "fragrance-free" in q or "sensitive" in q:
            return (
                "Both the CeraVe on the left and the Cetaphil on the right are fragrance-free and "
                "gentle for sensitive skin."
            )
        elif "dry skin" in q or "very dry" in q:
            return (
                "The CeraVe Moisturizing Cream on your left is the richest option, formulated with ceramides "
                "to restore the skin barrier."
            )
        elif "which one" in q or "recommend" in q:
            return (
                "If your skin is dry or sensitive, I recommend the CeraVe tub on the left. "
                "If you prefer a lightweight, oil-free gel, I recommend the Neutrogena jar in the center."
            )
        elif "pick" in q or "grab" in q or "guide" in q or "want" in q:
            # User wants to select
            if "cerave" in q:
                return "Target selected: CeraVe Moisturizing Cream. Preparing hand guidance."
            elif "neutrogena" in q:
                return "Target selected: Neutrogena Hydro Boost. Preparing hand guidance."
            elif "cetaphil" in q:
                return "Target selected: Cetaphil Daily Hydrating Lotion. Preparing hand guidance."
            else:
                return "Please tell me which product you want: CeraVe, Neutrogena, or Cetaphil."
        else:
            return (
                "Visible on this shelf are CeraVe Moisturizing Cream on the left, Neutrogena Hydro Boost in the center, "
                "and Cetaphil Daily Hydrating Lotion on the right. You can ask about skin types, ingredients, or say which one you'd like."
            )
