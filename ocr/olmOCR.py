import torch
import base64
import os
import re
import argparse

from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from pypdf import PdfReader

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}

def extract_date_for_filename(text: str) -> str:
    # Match: March 25, 1873
    m = re.search(r"\b([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})\b", text)
    if m:
        month_name, day, year = m.group(1), int(m.group(2)), int(m.group(3))
        month_num = MONTHS.get(month_name.lower())
        if month_num:
            return f"{year}_{month_num:02d}_{day:02d}.txt"

    # Fallback: year only
    y = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", text)
    if y:
        return f"{y.group(1)}.txt"

    return "UNKNOWN_DATE.txt"


# ---------- NEW (minimal): page-range args for parallelization ----------
parser = argparse.ArgumentParser()
parser.add_argument("--start_page", type=int, default=3, help="1-indexed start page (inclusive)")
parser.add_argument("--end_page", type=int, default=None, help="1-indexed end page (inclusive). Default: last page")
args = parser.parse_args()
# ----------------------------------------------------------------------


# Initialize the model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-2-7B-1025-FP8",
    device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Use local PDF (no urllib)
pdf_path = 

# Output directory
output_dir = 
os.makedirs(output_dir, exist_ok=True)

# Count pages
num_pages = len(PdfReader(pdf_path).pages)

# ---------- NEW (minimal): clamp requested range ----------
start_page = max(1, args.start_page)
end_page = num_pages if args.end_page is None else min(num_pages, args.end_page)
# --------------------------------------------------------


# Process selected page range
for page_num in range(start_page, end_page + 1):
    # Render page to base64 PNG
    image_base64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=1288)

    # Build the full prompt (add congressional speech guidance + full extraction instruction)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                {"type": "text", "text": "The document contains newline, and some names starting at the beginnning of a paragraph with all capitalized are the name of the speaker. Output the FULL text after the YAML."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]

    # Apply the chat template and processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}

    # Generate the output (deterministic OCR)
    output = model.generate(
        **inputs,
        temperature=0.0,
        max_new_tokens=8192,
        num_return_sequences=1,
        do_sample=False,
    )

    # Decode the output
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # Save OCR output (one file per page)
    base_filename = extract_date_for_filename(text_output[0])
    stem, _ = os.path.splitext(base_filename)
    filename = f"{stem}_p{page_num:04d}.txt"

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_output[0].strip() + "\n")

    print(f"Saved page {page_num}/{num_pages} -> {output_path}", flush=True)
