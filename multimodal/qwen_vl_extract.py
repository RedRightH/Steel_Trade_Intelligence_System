"""
qwen_vl_extract.py — Multimodal extraction using Qwen2-VL.

Extracts structured data from images of steel trade documents:
  - Anti-dumping investigation tables
  - Tariff schedule pages
  - Ministry of Steel report charts

Requires GPU (Colab T4 or better). Tested with:
  pip install transformers>=4.45.0 qwen-vl-utils pillow

Usage:
  python multimodal/qwen_vl_extract.py --image path/to/doc_page.png
  python multimodal/qwen_vl_extract.py --test   # run 3 built-in test cases

Model: Qwen/Qwen2-VL-7B-Instruct (7B, 4-bit, ~6GB VRAM on T4)
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ── Model + processor (lazy-loaded so import doesn't crash on CPU-only machines) ─
_model     = None
_processor = None

def _load_model():
    global _model, _processor
    if _model is not None:
        return

    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig

    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"[vl] Loading {MODEL_ID} in 4-bit …")
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model     = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print(f"[vl] Model ready. VRAM: {__import__('torch').cuda.memory_allocated()/1e9:.1f} GB")


def extract_from_image(image_path: str, prompt: str = None) -> dict:
    """
    Run Qwen2-VL on a document image and return extracted structured data.

    Args:
        image_path: Path to PNG/JPEG image of a document page.
        prompt:     Custom instruction. Defaults to generic document extraction prompt.

    Returns:
        dict with keys: raw_text, structured (dict or None), model, image_path
    """
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    _load_model()

    if prompt is None:
        prompt = (
            "This is a page from an Indian steel trade policy document. "
            "Extract all tabular data, numeric values, country names, product names, "
            "investigation numbers, and dates. "
            "Return the result as a structured JSON object with keys: "
            "tables (list of {headers, rows}), key_values (dict), entities (list of strings). "
            "If there are no tables, return an empty list for tables."
        )

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens=1024)

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, output_ids)
    ]
    raw_text = _processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    # Strip markdown code fences before JSON parsing (model wraps output in ```json ... ```)
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean.rstrip())

    structured = None
    # Try array first, then object
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        try:
            start = clean.find(start_char)
            end   = clean.rfind(end_char) + 1
            if start != -1 and end > start:
                structured = json.loads(clean[start:end])
                break
        except json.JSONDecodeError:
            continue

    # Partial-array recovery: if array was truncated (tokens ran out), salvage complete objects
    if structured is None and clean.find("[") != -1:
        try:
            # Find last complete `}` before any trailing incomplete content
            arr_start = clean.find("[")
            last_brace = clean.rfind("},")
            if last_brace == -1:
                last_brace = clean.rfind("}")
            if last_brace != -1:
                partial = clean[arr_start : last_brace + 1] + "]"
                structured = json.loads(partial)
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "image_path": str(image_path),
        "raw_text":   raw_text,
        "structured": structured,
        "model":      "Qwen/Qwen2-VL-2B-Instruct",
    }


# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name":   "AD_investigation_table",
        "desc":   "Anti-dumping investigation table (countries, dumping margins, duty rates)",
        "prompt": (
            "Extract the anti-dumping investigation table. "
            "Return JSON with: investigation_id, product, countries (list of {name, dumping_margin_pct, duty_pct}), "
            "date_of_initiation, investigating_authority."
        ),
        "image":  None,  # set to actual path when running
    },
    {
        "name":   "tariff_schedule_page",
        "desc":   "Tariff schedule page (HS codes, basic customs duty, IGST rates)",
        "prompt": (
            "Extract all tariff line items from this page. "
            "Return JSON with: items (list of {hs_code, description, bcd_pct, igst_pct, total_incidence_pct})."
        ),
        "image":  None,
    },
    {
        "name":   "steel_production_chart",
        "desc":   "Bar/line chart showing India steel production by FY",
        "prompt": (
            "This chart shows Indian steel production data. "
            "Extract all data points. "
            "Return JSON with: chart_title, x_axis_label, y_axis_label, y_unit, "
            "data_points (list of {year, value})."
        ),
        "image":  None,
    },
]


def run_tests(image_dir: str = None):
    """
    Run the 3 test cases.
    If image_dir is provided, looks for images named by test case name (e.g., AD_investigation_table.png).
    Otherwise creates synthetic placeholder images to verify the pipeline runs end-to-end.
    """
    from PIL import Image, ImageDraw, ImageFont
    import tempfile, os

    results = []
    for tc in TEST_CASES:
        print(f"\n=== Test: {tc['name']} ===")
        print(f"    {tc['desc']}")

        if image_dir and tc["image"] is None:
            candidates = [
                Path(image_dir) / f"{tc['name']}.png",
                Path(image_dir) / f"{tc['name']}.jpg",
            ]
            for c in candidates:
                if c.exists():
                    tc["image"] = str(c)
                    break

        if tc["image"] is None or not Path(tc["image"]).exists():
            # Create a synthetic test image
            img = Image.new("RGB", (800, 600), color="white")
            d   = ImageDraw.Draw(img)
            d.text((20, 20),  "INDIA STEEL TRADE INTELLIGENCE PLATFORM", fill="black")
            d.text((20, 60),  f"Test document: {tc['name']}", fill="black")
            d.text((20, 100), "Country | Dumping Margin | AD Duty", fill="black")
            d.text((20, 140), "China   |     65.3%       |  18.2%", fill="black")
            d.text((20, 180), "Korea   |     22.1%       |   5.4%", fill="black")
            d.text((20, 220), "Japan   |     19.8%       |   4.9%", fill="black")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img.save(f.name)
                tc["image"] = f.name
            print(f"    [synthetic image created at {tc['image']}]")

        result = extract_from_image(tc["image"], tc["prompt"])
        result["test_case"] = tc["name"]
        results.append(result)

        print(f"    Raw output (first 300 chars): {result['raw_text'][:300]}")
        if result["structured"] is not None:
            s = result["structured"]
            if isinstance(s, list):
                print(f"    Parsed JSON: list of {len(s)} item(s), first keys: {list(s[0].keys()) if s else []}")
            else:
                print(f"    Parsed JSON keys: {list(s.keys())}")
        else:
            print("    Could not parse JSON from output (raw text returned)")

    # Save results
    out_path = Path(__file__).parent / "vl_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2-VL document extraction")
    parser.add_argument("--image",     help="Path to image file")
    parser.add_argument("--prompt",    help="Custom extraction prompt")
    parser.add_argument("--test",      action="store_true", help="Run 3 built-in test cases")
    parser.add_argument("--image-dir", help="Directory with test images (for --test)")
    args = parser.parse_args()

    if args.test:
        run_tests(args.image_dir)
    elif args.image:
        result = extract_from_image(args.image, args.prompt)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
        sys.exit(1)
