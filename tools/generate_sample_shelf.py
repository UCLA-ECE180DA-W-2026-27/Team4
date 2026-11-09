#!/usr/bin/env python3
"""Generate a realistic test image of a CVS skincare shelf.

Creates data/sample_cvs_shelf.jpg containing CeraVe, Neutrogena, and Cetaphil
skincare products for offline testing, demos, and VLM evaluation.
"""

from pathlib import Path
import cv2
import numpy as np


def generate_cvs_shelf_image(output_path: str = "data/sample_cvs_shelf.jpg", width: int = 1280, height: int = 800):
    """Render a synthetic 1280x800 retail shelf image."""
    img = np.full((height, width, 3), (235, 235, 240), dtype=np.uint8)

    # 1. Background wall and shadows
    cv2.rectangle(img, (0, 0), (width, height), (220, 222, 225), -1)

    # 2. Metal shelf racks (horizontal boards)
    shelf_y = int(height * 0.75)
    shelf_thickness = 35

    # Shelf shadow
    cv2.rectangle(img, (0, shelf_y + shelf_thickness), (width, shelf_y + shelf_thickness + 20), (180, 180, 185), -1)
    # Shelf body
    cv2.rectangle(img, (0, shelf_y), (width, shelf_y + shelf_thickness), (150, 155, 160), -1)
    # Front rail
    cv2.rectangle(img, (0, shelf_y + 5), (width, shelf_y + shelf_thickness - 5), (200, 205, 210), -1)

    # Price tags along the rail
    tag_coords = [int(width * 0.22), int(width * 0.52), int(width * 0.77)]
    tag_labels = ["$17.99\nCeraVe", "$21.49\nNeutrogena", "$16.49\nCetaphil"]
    for x_tag, label in zip(tag_coords, tag_labels):
        cv2.rectangle(img, (x_tag - 45, shelf_y + 8), (x_tag + 45, shelf_y + 28), (255, 255, 255), -1)
        cv2.rectangle(img, (x_tag - 45, shelf_y + 8), (x_tag + 45, shelf_y + 28), (0, 0, 180), 1)
        cv2.putText(img, label.split("\n")[0], (x_tag - 38, shelf_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1)

    # 3. Product 1: CeraVe Moisturizing Cream (Left: ~380-720 y, ~120-380 x)
    x1_c, y1_c = int(width * 0.13), int(height * 0.40)
    x2_c, y2_c = int(width * 0.36), shelf_y

    # White tub body
    cv2.rectangle(img, (x1_c, y1_c), (x2_c, y2_c), (250, 250, 250), -1)
    cv2.rectangle(img, (x1_c, y1_c), (x2_c, y2_c), (180, 180, 180), 2)
    # Dark blue band
    cv2.rectangle(img, (x1_c, y1_c + 20), (x2_c, y1_c + 70), (140, 50, 10), -1)  # BGR
    # Text
    cv2.putText(img, "CeraVe", (x1_c + 35, y1_c + 58), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
    cv2.putText(img, "Moisturizing Cream", (x1_c + 20, y1_c + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
    cv2.putText(img, "For Normal to Dry Skin", (x1_c + 20, y1_c + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
    cv2.putText(img, "with 3 Essential Ceramides", (x1_c + 20, y1_c + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 50, 10), 1)
    cv2.putText(img, "16 OZ / 453 g", (x1_c + 85, y2_c - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # 4. Product 2: Neutrogena Hydro Boost Water Gel (Center: ~340-680 y, ~420-640 x)
    x1_n, y1_n = int(width * 0.43), int(height * 0.43)
    x2_n, y2_n = int(width * 0.63), shelf_y

    # Vibrant blue circular jar body
    cv2.rectangle(img, (x1_n, y1_n), (x2_n, y2_n), (210, 130, 20), -1)  # BGR deep sky blue
    cv2.rectangle(img, (x1_n, y1_n), (x2_n, y2_n), (180, 90, 10), 2)
    # Silver jar lid
    cv2.rectangle(img, (x1_n - 5, y1_n - 25), (x2_n + 5, y1_n), (220, 220, 225), -1)
    cv2.rectangle(img, (x1_n - 5, y1_n - 25), (x2_n + 5, y1_n), (160, 160, 165), 1)
    # Text
    cv2.putText(img, "Neutrogena", (x1_n + 25, y1_n + 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, "Hydro Boost", (x1_n + 25, y1_n + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(img, "Water Gel", (x1_n + 45, y1_n + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 255), 1)
    cv2.putText(img, "Hyaluronic Acid", (x1_n + 35, y1_n + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "Oil-Free", (x1_n + 65, y2_n - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # 5. Product 3: Cetaphil Daily Hydrating Lotion (Right: ~360-740 y, ~680-890 x)
    x1_ct, y1_ct = int(width * 0.70), int(height * 0.38)
    x2_ct, y2_ct = int(width * 0.88), shelf_y

    # White slim pump bottle
    cv2.rectangle(img, (x1_ct, y1_ct), (x2_ct, y2_ct), (248, 248, 252), -1)
    cv2.rectangle(img, (x1_ct, y1_ct), (x2_ct, y2_ct), (190, 190, 195), 2)
    # Green pump top
    cv2.rectangle(img, (x1_ct + 40, y1_ct - 40), (x2_ct - 40, y1_ct), (80, 160, 50), -1)
    # Text
    cv2.putText(img, "Cetaphil", (x1_ct + 30, y1_ct + 65), cv2.FONT_HERSHEY_DUPLEX, 1.1, (160, 80, 20), 2)
    cv2.putText(img, "Daily Hydrating", (x1_ct + 22, y1_ct + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(img, "Lotion", (x1_ct + 55, y1_ct + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(img, "With Hyaluronic Acid", (x1_ct + 22, y1_ct + 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 160, 50), 1)
    cv2.putText(img, "Fragrance Free", (x1_ct + 35, y1_ct + 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)

    # Save to disk
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[✓] Generated synthetic CVS shelf image: {out_path.resolve()} ({width}x{height})")


if __name__ == "__main__":
    generate_cvs_shelf_image()

