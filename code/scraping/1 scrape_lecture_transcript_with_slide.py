# %%


# %%
import re
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

"""
lecture_slide_utils.py  (v3 — PyMuPDF only)
=============================================
Two standalone utility functions for NPTEL-style lecture PDFs.

Only dependency: PyMuPDF (fitz)  — no pdfplumber, no OpenCV, no Pillow needed.

    pip install PyMuPDF

FUNCTIONS
---------
1. extract_slide_images(pdf_path)
   - Uses fitz page.get_images() + doc.extract_image() to pull embedded images
     directly from the PDF binary (no visual box guessing).
   - Saves to <stem>_slides/ folder, returns list of paths.

2. get_slide_transcripts(pdf_path)
   - Uses fitz page.get_text("words") to get word-level bounding boxes,
     reconstructs lines with proper spacing (fixes "fromthe" / "ofview").
   - Splits on "(Refer Slide Time: MM:SS)" markers, returns list of dicts.
"""

import io
import os
import re
from pathlib import Path

import fitz   # PyMuPDF — only dependency


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 1 — extract_slide_images
# ══════════════════════════════════════════════════════════════════════════════

def extract_slide_images(pdf_path, dpi=150, min_width=300, min_height=200):
    """
    Extract every slide image from a lecture PDF using PyMuPDF's native
    image extraction (fitz page.get_images + doc.extract_image).

    Reads actual embedded image XObjects from the PDF binary — no OpenCV,
    no contour guessing. Works correctly for NPTEL-style PDFs.

    Filters applied to skip non-slide images
    -----------------------------------------
    - min_width / min_height : ignore tiny logos/icons (default 300x200 px)
    - small-square filter    : skip professor webcam thumbnail
      (width < 500 AND height < 500 AND aspect ratio near 1:1)
    - duplicate xref guard   : same image on multiple pages saved only once

    Folder structure
    ----------------
    Input : lec7.pdf
    Output: lec7_slides/
                lec7_img1.png
                lec7_img2.png  ...

    Returns
    -------
    list of str — sorted paths to saved PNG files
    """
    pdf_path  = str(pdf_path)
    stem      = Path(pdf_path).stem
    slides_dir = os.path.join(
        os.path.dirname(os.path.abspath(pdf_path)), f"{stem}_slides"
    )
    os.makedirs(slides_dir, exist_ok=True)

    doc         = fitz.open(pdf_path)
    saved_paths = []
    seen_xrefs  = set()
    img_counter = 1

    for page_num in range(len(doc)):
        page = doc[page_num]

        # get_images(full=True) -> list of tuples:
        # (xref, smask, width, height, bpc, colorspace, alt_cs, name, filter, referencer)
        for img_info in page.get_images(full=True):
            xref       = img_info[0]
            img_width  = img_info[2]
            img_height = img_info[3]

            # Skip duplicates
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            # Filter 1: skip tiny decorative images
            if img_width < min_width or img_height < min_height:
                continue

            # Filter 2: skip small near-square thumbnails (professor cam)
            aspect = img_width / img_height if img_height > 0 else 0
            if img_width < 500 and img_height < 500 and 0.6 < aspect < 1.6:
                continue

            try:
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                img_ext    = base_image["ext"]   # "png", "jpeg", etc.

                # Save via fitz.Pixmap for format normalisation (no Pillow needed)
                pix      = fitz.Pixmap(img_bytes)
                # Convert CMYK or other colorspaces to RGB
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                out_path = os.path.join(slides_dir, f"{stem}_img{img_counter}.png")
                pix.save(out_path)
                saved_paths.append(out_path)
                print(f"  Saved [{img_width}x{img_height}] -> {out_path}")
                img_counter += 1
            except Exception as exc:
                print(f"  Skipped xref {xref}: {exc}")

    doc.close()

    if not saved_paths:
        print("  No embedded images found — using page-render fallback...")
        saved_paths = _fallback_render_pages(pdf_path, stem, slides_dir, dpi)

    print(f"\nFolder : {slides_dir}")
    print(f"Slides : {len(saved_paths)}")
    return sorted(saved_paths)


def _fallback_render_pages(pdf_path, stem, slides_dir, dpi):
    """Render each full page as PNG. Only used when PDF has no embedded images."""
    scale = dpi / 72
    doc   = fitz.open(pdf_path)
    paths = []
    for page_num in range(len(doc)):
        pix      = doc[page_num].get_pixmap(matrix=fitz.Matrix(scale, scale))
        out_path = os.path.join(slides_dir, f"{stem}_page{page_num + 1}.png")
        pix.save(out_path)
        paths.append(out_path)
    doc.close()
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 2 — get_slide_transcripts
# ══════════════════════════════════════════════════════════════════════════════

def _extract_clean_text(pdf_path):
    """
    Extract full text with correct word spacing using fitz page.get_text("words").

    Root cause of "fromthe" / "ofview" with get_text("text"):
        PyMuPDF's plain-text extractor joins character spans without checking
        horizontal gaps, so adjacent words from different spans get merged.

    Fix — use word-level extraction:
        get_text("words") returns one entry per word:
        (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        Words are already separated; we just need to reconstruct line order
        and join with spaces.

    Algorithm
    ---------
    1. get_text("words") -> list of word tuples, sorted by (y0, x0)
    2. Group words whose top-edge (y0) values are within LINE_TOL px
       -> same line
    3. Within each line sort by x0, join with " "
    4. Join lines with newline
    """
    LINE_TOL  = 5   # px — words within this vertical range share a line
    all_lines = []

    doc = fitz.open(pdf_path)
    for page in doc:
        # Each word: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        words = page.get_text("words", sort=True)   # sort=True -> reading order
        if not words:
            continue

        lines        = []
        current_line = [words[0]]

        for word in words[1:]:
            y0_prev = current_line[-1][1]   # top of last word in current line
            y0_curr = word[1]               # top of this word
            if abs(y0_curr - y0_prev) <= LINE_TOL:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        lines.append(current_line)

        for line in lines:
            line.sort(key=lambda w: w[0])   # sort by x0
            all_lines.append(" ".join(w[4] for w in line))

    doc.close()
    return "\n".join(all_lines)


def get_slide_transcripts(pdf_path):
    """
    Split the lecture PDF transcript on "(Refer Slide Time: MM:SS)" markers.

    Uses fitz get_text("words") for clean word spacing — no pdfplumber needed.

    Returns
    -------
    list of dict, each with:
        slide_index (int)  0 = intro before first marker; 1, 2, 3... = slides
        marker      (str)  "(Refer Slide Time: MM:SS)"  or  None for intro
        transcript  (str)  clean, properly spaced text for that segment

    Example
    -------
    segs = get_slide_transcripts("lec7.pdf")
    print(segs[0]["transcript"])         # intro text
    print(segs[1]["marker"])             # (Refer Slide Time: 00:32)
    print(segs[2]["transcript"][:200])   # slide 2 explanation
    """
    pdf_path  = str(pdf_path)
    marker_re = re.compile(r"\(Refer\s+Slide\s+Time:\s*\d{2}:\d{2}\)")

    full_text     = _extract_clean_text(pdf_path)
    markers       = marker_re.findall(full_text)
    segments_text = marker_re.split(full_text)   # [0]=intro, [1..n]=per marker

    result = []
    for i, segment in enumerate(segments_text):
        result.append({
            "slide_index": i,
            "marker":      markers[i - 1] if i > 0 else None,
            "transcript":  segment.strip(),
        })

    print(f"PDF           : {Path(pdf_path).name}")
    print(f"Markers found : {len(markers)}")
    print(f"Segments      : {len(result)}  (0=intro, 1-{len(result)-1}=slides)")
    return result





# %%
import os
from google import genai
from google.genai import types

# Initialize client once (outside loop ideally)
client = genai.Client(
    vertexai=True,
    api_key=API_KEY
)

def ocr_slides_with_gemini(images):
    """
    images: list of either
        - file paths (str), OR
        - PIL Images

    returns: list of OCR text (one per image)
    """
    ocr_results = []

    for img in images:
        # Convert image to bytes
        if isinstance(img, str):  # file path
            with open(img, "rb") as f:
                image_bytes = f.read()
            mime_type = "image/png" if img.endswith(".png") else "image/jpeg"

        else:  # PIL Image
            import io
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            mime_type = "image/png"

        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",  # 👈 vision model
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type
                    ),
                    """Extract all readable content from this slide and return it in a structured format.

Follow these rules strictly:

1. Identify and label content using the following tags:
   - <title>: main slide title
   - <heading>: section headings
   - <subheading>: subsection headings
   - <bullet>: bullet point text (one per line)
   - <text>: any normal paragraph text
   - <table>: if a table is present, summarize its contents clearly
   - <diagram>: if a diagram/figure is present, describe it briefly

2. Preserve the logical structure and hierarchy of the slide.

3. Do NOT include decorative elements, backgrounds, or irrelevant graphics.

4. If text is unclear, make a best effort to reconstruct it.

5. Output should be clean, readable, and properly formatted with tags.

6. Do not leave any text

Example format:

<title>Introduction to Neural Networks</title>
<heading>What is a Neural Network?</heading>
<bullet>Computational model inspired by the brain</bullet>
<bullet>Composed of layers of neurons</bullet>
<diagram>Simple feedforward network with input, hidden, and output layers</diagram>"""
                ]
            )

            text = response.text.strip() if response.text else ""
        except Exception as e:
            print(f"Error processing image: {e}")
            text = ""

        ocr_results.append(text)

    return ocr_results

import re

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# %%
import time
def ocr_slides_with_gemini_batched(images, batch_size=5,max_retries=3, sleep_time=2):
    """
    Process slides in batches to reduce API calls
    """
    import io

    ocr_results = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]

        parts = []
        prompt = "You are given multiple slide images.\n\n"

        # Add images with numbering
        for idx, img in enumerate(batch):
            if isinstance(img, str):
                with open(img, "rb") as f:
                    image_bytes = f.read()
                mime_type = "image/png" if img.endswith(".png") else "image/jpeg"
            else:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                mime_type = "image/png"

            parts.append(types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            ))

            prompt += f"\nSlide {idx+1}:\n"

        # 🔥 Strong structured prompt
        prompt += """
For EACH slide, extract content separately.

Follow these guidelines for each slide:
Extract all readable content from this slide and return it in a structured format.

Follow these rules strictly:

1. Identify and label content using the following tags:
   - <title>: main slide title
   - <heading>: section headings
   - <subheading>: subsection headings
   - <bullet>: bullet point text (one per line)
   - <text>: any normal paragraph text
   - <table>: if a table is present, summarize its contents clearly
   - <diagram>: if a diagram/figure is present, describe it briefly

2. Preserve the logical structure and hierarchy of the slide.

3. Do NOT include decorative elements, backgrounds, or irrelevant graphics.

4. If text is unclear, make a best effort to reconstruct it.

5. Output should be clean, readable, and properly formatted with tags.

6. Do not leave any text

Example format:

<title>Introduction to Neural Networks</title>
<heading>What is a Neural Network?</heading>
<bullet>Computational model inspired by the brain</bullet>
<bullet>Composed of layers of neurons</bullet>
<diagram>Simple feedforward network with input, hidden, and output layers</diagram>


Return output in this EXACT format:

=== SLIDE 1 ===
<title>Introduction to Neural Networks</title>
<heading>What is a Neural Network?</heading>
<bullet>Computational model inspired by the brain</bullet>
<bullet>Composed of layers of neurons</bullet>
<diagram>Simple feedforward network with input, hidden, and output layers</diagram>

=== SLIDE 2 ===
<title>Introduction to Neural Networks</title>
<heading>What is a Neural Network?</heading>
<bullet>Computational model inspired by the brain</bullet>
<bullet>Composed of layers of neurons</bullet>
<diagram>Simple feedforward network with input, hidden, and output layers</diagram>

Rules:
- Do NOT merge slides
- Maintain strict separation
- Use tags: title, heading, subheading, bullet, text, table, diagram
- Do not skip any visible text
"""

        parts.append(prompt)

         # 🔁 RETRY LOGIC
        batch_text = ""
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=parts
                )

                batch_text = response.text if response.text else ""
                break  # success

            except Exception as e:
                print(f"⚠️ Batch failed (attempt {attempt+1}): {e}")
                time.sleep(sleep_time)

        # ❌ If still failed after retries
        if not batch_text:
            print("❌ Batch completely failed")
            ocr_results.extend(["NOT_EXTRACTED"] * len(batch))
            continue

        # 🔍 Split output
        slides_output = batch_text.split("=== SLIDE")
        slides_output = [s.strip() for s in slides_output if s.strip()]
        slides_output = [s.split("===", 1)[-1].strip() for s in slides_output]

        # ⚠️ Handle mismatch
        if len(slides_output) != len(batch):
            print("⚠️ Output mismatch, filling fallback")
            slides_output = slides_output + ["NOT_EXTRACTED"] * (len(batch) - len(slides_output))

        ocr_results.extend(slides_output)

        # ⏳ Sleep between batches
        time.sleep(sleep_time)


    return ocr_results

# %%
import os
import pandas as pd

output_dir = "/home/roy.2/rehtorical_segmentation_ai_Course/ai_df_lecture_transcript"
os.makedirs(output_dir, exist_ok=True)


if __name__ == "__main__":

    pdf_folder = "/home/roy.2/rehtorical_segmentation_ai_Course/lecture_pdf"   # 👈 folder containing PDFs
    base_output_dir = "./slide_outputs"
    os.makedirs(base_output_dir, exist_ok=True)

    all_images = []
    all_ocr_output = []
    all_transcripts = []
    all_lecture_numbers = []
    all_slide_numbers = []

    pdf_files = sorted(
    [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")],
    key=extract_number
    )

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(pdf_path)  # replace with your processing logic

        lecture_num = extract_number(pdf_file)  # 👈 get lecture number

        images = extract_slide_images(pdf_path)
        segs = get_slide_transcripts(pdf_path)



        # Align lengths
        if len(segs) > len(images):
            segs = segs[len(segs) - len(images):]
        elif len(images) > len(segs):
            images = images[len(images) - len(segs):]

        transcripts = [s.get('transcript') for s in segs]

        # 🔥 OCR from slides
        ocr_texts = ocr_slides_with_gemini_batched(images)

        min_len = min(len(images), len(segs))
        if len(ocr_texts) != min_len:
            print(f"⚠️ Fixing mismatch in {pdf_file}")
            ocr_texts = ocr_texts + ["NOT_EXTRACTED"] * (min_len - len(ocr_texts))

        # STEP 4: local storage (per lecture)
        lecture_data = []

        for i, (img, ocr, transcript) in enumerate(zip(images, ocr_texts, transcripts), start=1):
            lecture_data.append({
                "lecture_number": lecture_num,
                "slide_number": i,
                "ocr_text": ocr,
                "transcript": transcript
            })

        # ✅ STEP 5: SAVE THIS LECTURE (incremental save)
        lecture_df = pd.DataFrame(lecture_data)

        lecture_file = os.path.join(
            "/home/roy.2/rehtorical_segmentation_ai_Course/ai_df_lecture_transcript",
            f"lecture_{lecture_num}.csv"
        )

        lecture_df.to_csv(lecture_file, index=False)
        print(f"✅ Saved lecture file: {lecture_file}")


        # STEP 6: extend global lists (for final merge)
        for row in lecture_data:
            all_images.append(None)  # optional
            all_ocr_output.append(row["ocr_text"])
            all_transcripts.append(row["transcript"])
            all_slide_numbers.append(row["slide_number"])
            all_lecture_numbers.append(row["lecture_number"])



# %%
final_df = pd.DataFrame({
    "lecture_number": all_lecture_numbers,
    "slide_number": all_slide_numbers,
    "ocr_text": all_ocr_output,
    "transcript": all_transcripts
})

final_file = os.path.join(
    "/home/roy.2/rehtorical_segmentation_ai_Course/ai_df_lecture_transcript",
    "all_lectures.csv"
)

final_df.to_csv(final_file, index=False)
print(f"🔥 Saved final merged file: {final_file}")

# %%



