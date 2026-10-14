import re
import sys
from PIL import Image
from google import genai
from google.genai import types

def complete_sketch_with_gemini(image_path: str, output_svg_path: str = "added_lines.svg"):
    # 1. Load the image and detect dimensions
    img = Image.open(image_path)
    width, height = img.size

    # 2. Read image bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # 3. Initialize Gemini client
    client = genai.Client()

    prompt = f"""
    The input is an incomplete simple line drawing with dimensions {width}x{height}.

    Tasks:
    1. Identify what this sketch is depicting (e.g., a face, character, animal, object).
    2. Identify the key missing features needed to finish the sketch.
    3. Generate the vector lines/strokes to complete the drawing.

    Strict Output Requirements:
    - Return ONLY valid SVG markup representing ONLY THE ADDED LINES (do NOT duplicate or redraw the existing lines).
    - Set viewBox="0 0 {width} {height}", width="{width}", and height="{height}".
    - Match the hand-drawn stroke style: stroke="black", stroke-width="4" to "6", fill="none", stroke-linecap="round", stroke-linejoin="round".
    - Use `<path>` elements with organic curves (C, S, or Q commands) so they blend naturally with the sketch.
    - Output raw SVG directly. Do not add markdown code fences, headers, or explanations.
    """

    # 4. Generate the completion using Gemini 2.5 Flash
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ],
    )

    content = response.text or ""

    # 5. Extract SVG markup if enclosed in markdown code fences
    svg_match = re.search(r"<svg[\s\S]*?<\/svg>", content)
    svg_content = svg_match.group(0) if svg_match else content.strip()

    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    print(f"Added lines successfully saved to: {output_svg_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python complete_sketch_gemini.py <input_image.png> [output.svg]")
        sys.exit(1)

    input_img = sys.argv[1]
    out_svg = sys.argv[2] if len(sys.argv) > 2 else "added_lines.svg"
    complete_sketch_with_gemini(input_img, out_svg)