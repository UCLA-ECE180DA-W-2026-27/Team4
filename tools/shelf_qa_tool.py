#!/usr/bin/env python3
"""Interactive Shelf Inspection & Conversational Q&A Tool.

Inspects retail shelf images using Google Gemini Flash, extracts products and
bounding boxes for visual servoing, and conducts natural multi-turn dialogue.

Usage:
    # Offline / simulated test with sample shelf photo:
    python tools/shelf_qa_tool.py --image data/sample_cvs_shelf.jpg

    # Live call with your Gemini API Key:
    python tools/shelf_qa_tool.py --image data/sample_cvs_shelf.jpg --api-key YOUR_API_KEY

    # Capture from live camera:
    python tools/shelf_qa_tool.py --camera 0
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from vision import ShelfAnalyzer


def capture_frame_from_camera(camera_index: int = 0, output_path: str = "data/live_shelf.jpg") -> Path:
    """Capture a single frame from an attached camera."""
    print(f"\n[+] Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video camera index {camera_index}")

    # Allow camera to auto-expose
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to read image frame from camera.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"[✓] Captured snapshot saved to: {out.resolve()}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Gemini Flash Shelf Inspection and Conversational Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        default="data/sample_cvs_shelf.jpg",
        help="Path to shelf image file",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device index to capture live image",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google Gemini API Key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--output",
        default="data/shelf_detected.jpg",
        help="Path to save annotated detection image",
    )
    parser.add_argument(
        "--question",
        "-q",
        default=None,
        help="Optional single question to ask non-interactively",
    )

    args = parser.parse_args()

    # Determine image source
    if args.camera is not None:
        image_path = capture_frame_from_camera(args.camera)
    else:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"[!] Image not found at {image_path}. Generating default sample...")
            from tools.generate_sample_shelf import generate_cvs_shelf_image
            generate_cvs_shelf_image(str(image_path))

    # Initialize Shelf Analyzer
    analyzer = ShelfAnalyzer(api_key=args.api_key)

    print("\n" + "=" * 65)
    print("       CVS SHELF EXPLORATION & MULTIMODAL Q&A ENGINE")
    print("=" * 65)
    print(f"Image Source : {image_path}")
    mode_desc = "LIVE Google Gemini Flash" if (analyzer.client.client and analyzer.client.api_key) else "SIMULATION / OFFLINE MODE"
    print(f"Engine Mode  : {mode_desc}")
    print("-" * 65)

    print("\n[+] Scanning shelf photo with Gemini Flash...")
    inspection = analyzer.inspect_shelf(image_path)

    # 1. Spoken Summary for blind user
    print("\n[Spoken Audio Description for Blind Shopper]:")
    print(f"  🔊 \"{inspection.spoken_summary}\"\n")

    # 2. Product Table
    print("-" * 65)
    print(f"{'ID':<4} | {'Product Name':<32} | {'Side':<7} | {'Level':<10} | {'Box [ymin, xmin, ymax, xmax]'}")
    print("-" * 65)
    for prod in inspection.products:
        box_str = f"[{prod.box_2d[0]}, {prod.box_2d[1]}, {prod.box_2d[2]}, {prod.box_2d[3]}]"
        print(f"{prod.product_id.upper():<4} | {prod.name:<32} | {prod.side.title():<7} | {prod.shelf_level.title():<10} | {box_str}")
    print("-" * 65)

    # 3. Save initial detection overlay
    analyzer.render_annotated_image(args.output)
    print(f"[✓] Annotated detection image saved to: {Path(args.output).resolve()}")

    # 4. Handle non-interactive single question
    if args.question:
        print(f"\nUser Query: \"{args.question}\"")
        answer = analyzer.ask(args.question)
        print(f"🔊 Answer : \"{answer}\"\n")
        if analyzer.selected_target:
            pixel_box = analyzer.get_target_pixel_box()
            print(f"[★] Target Locked: {analyzer.selected_target.name}")
            print(f"    Pixel Coordinates (x1, y1, x2, y2): {pixel_box}")
            analyzer.render_annotated_image(args.output)
            print(f"[✓] Updated target overlay saved to: {Path(args.output).resolve()}")
        return

    # 5. Interactive Conversational Dialogue Loop
    print("\n[Interactive Q&A Session Started]")
    print("You can ask questions about ingredients, skin types, or say which item you want.")
    print("Type 'exit' or 'q' to end session.")
    print("-" * 65)

    while True:
        try:
            user_input = input("\nShopper (voice/text) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEnding session.")
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            print("Session ended.")
            break

        answer = analyzer.ask(user_input)
        print(f"\nAssistant 🔊: \"{answer}\"")

        # If a target was selected by user
        if analyzer.selected_target:
            pixel_box = analyzer.get_target_pixel_box()
            print(f"\n[★] Visual Servoing Target Acquired:")
            print(f"    Product : {analyzer.selected_target.name}")
            print(f"    Side    : {analyzer.selected_target.side.upper()}")
            print(f"    Level   : {analyzer.selected_target.shelf_level.upper()}")
            print(f"    Pixels  : (x1={pixel_box[0]}, y1={pixel_box[1]}, x2={pixel_box[2]}, y2={pixel_box[3]})")
            # Refresh visualization with target reticle
            analyzer.render_annotated_image(args.output)
            print(f"[✓] Updated target overlay saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

