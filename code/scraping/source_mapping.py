# %%
import pandas as pd

df=pd.read_csv('/home/roy.2/rehtorical_segmentation_ai_Course/ai_df_lecture_transcript/segment_all_lectures.csv')

df.head(10)

import re
import ast

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# =========================
# COMPLETE PIPELINE (ONE CELL)
# =========================

import pandas as pd
import re
import ast
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

for i in range(len(df)):

    # =========================
    # 1. CLEAN TRANSCRIPT
    # =========================
    transcript = df.iloc[i]['transcript']
    transcript = re.sub(r"\s+", " ", str(transcript)).strip()

    # =========================
    # 2. SENTENCE TOKENIZATION
    # =========================
    sentences = sent_tokenize(transcript)

    # create s1, s2, ...
    sentence_dict = {
        f"s{j+1}": sent for j, sent in enumerate(sentences)
    }

    max_sent = len(sentence_dict)

    # =========================
    # 3. LOAD SEGMENT INFO
    # =========================
    segment_info = df.iloc[i]['segment_info']

    if isinstance(segment_info, str):
        segment_info = ast.literal_eval(segment_info)

    # =========================
    # 4. RECONSTRUCT TEXT
    # =========================
    for segment in segment_info:

        start = int(segment["start"][1:])
        end = int(segment["end"][1:])

        # ⚠️ check out-of-range
        if end > max_sent:
            print(f"⚠️ Row {i}: Out of range ({start}-{end}) > max={max_sent}")

        # generate keys
        keys = [f"s{k}" for k in range(start, end + 1)]

        # ⚠️ check missing
        missing = [k for k in keys if k not in sentence_dict]
        if missing:
            print(f"⚠️ Row {i}: Missing keys {missing}")

        # reconstruct text
        segment["text"] = " ".join(
            [sentence_dict[k] for k in keys if k in sentence_dict]
        )

    # =========================
    # 5. SAVE BACK TO DF
    # =========================
    df.at[i, 'segment_info'] = segment_info

    # =========================
    # 6. DEBUG PRINT (REMOVE LATER)
    # =========================
    if i == 0:
        print("\nSample Output:\n")
        print(segment_info)
   
    

# %% [markdown]
# parse book 

# %%


# ====== IMPORTS ======
import fitz  # PyMuPDF

# ====== STEP 1: EXTRACT ALL PAGES ======
def extract_pages(pdf_path):
    pages = {}

    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages[i] = text if text else ""

    return pages


# ====== STEP 2: FIND "Contents" START ======
def find_contents_start(pages):
    for i in range(1, len(pages) + 1):
        text = pages[i]

        if "Contents" in text:
            return i

    return None


# ====== STEP 3: FIND "Bibliography" END ======
def find_bibliography_end(pages, start_page):
    for i in range(start_page, len(pages) + 1):
        text = pages[i]

        if "Bibliography" in text:
            return i

    return None


# ====== STEP 4: EXTRACT TOC REGION ======
def extract_toc_lines(pages, start_page, end_page):
    toc_lines = []

    for i in range(start_page, end_page + 1):
        text = pages[i]

        # split into lines
        lines = text.split("\n")

        toc_lines.extend(lines)

    return toc_lines


# ====== RUN PIPELINE ======
pdf_path = "/home/roy.2/narrative_pipeline/AI course pdf.pdf"

# 1. extract all pages
pages = extract_pages(pdf_path)

# 2. find TOC start
start_page = find_contents_start(pages)


# 3. find TOC end
end_page = find_bibliography_end(pages, start_page)


# 4. extract TOC lines
toc_lines = extract_toc_lines(pages, start_page, end_page)

import re

def parse_toc_structured(toc_lines):
    """
    Handles the actual token-per-element structure:
        '1.1'
        'What Is AI?'
        '. . . . . .'
        '1'
    """
    chapters = []
    sections = []

    dot_line    = re.compile(r'^[.\s]+$')
    roman_skip  = re.compile(r'^(Contents|Bibliography|Index|[ivxlcdmIVXLCDM]+)$')
    subsec_num  = re.compile(r'^(\d+)\.(\d+)$')          # 1.1, 18.10
    chapter_num = re.compile(r'^(\d+)$')                  # 1, 2, 10 ...
    # inline: "25.7  Robotic Software Architectures . . . 1003"  (all in one token)
    inline_full = re.compile(r'^(\d+(?:\.\d+)?)\s+(.+?)\s+(\d+)\s*$')

    # Clean: drop empty strings
    tokens = [t.strip() for t in toc_lines if t.strip()]

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # ── skip noise ──────────────────────────────────────────────────
        if roman_skip.match(tok) or dot_line.match(tok):
            i += 1
            continue

        # ── Case A: entire entry on one token  "25.7 Title ... 1003" ───
        m_inline = inline_full.match(tok)
        if m_inline:
            num_str = m_inline.group(1)
            title   = re.sub(r'\s*[\.\s]{3,}.*$', '', m_inline.group(2)).strip()
            page    = int(m_inline.group(3))
            if '.' in num_str:
                sections.append({'num': num_str, 'title': title,
                                  'page': page, 'chapter': num_str.split('.')[0]})
            else:
                chapters.append({'num': num_str, 'title': title, 'page': page})
            i += 1
            continue

        # ── Case B: section number token  e.g. "1.1" ────────────────────
        m_sub = subsec_num.match(tok)
        if m_sub:
            num_str     = tok                        # "1.1"
            chapter_str = m_sub.group(1)             # "1"

            # next non-dot token = title
            j = i + 1
            title = ""
            while j < len(tokens) and dot_line.match(tokens[j]):
                j += 1
            if j < len(tokens) and not re.match(r'^\d+$', tokens[j]):
                title = re.sub(r'\s*[\.\s]{3,}.*$', '', tokens[j]).strip()
                j += 1

            # skip dot tokens to find page number
            while j < len(tokens) and dot_line.match(tokens[j]):
                j += 1

            # also handle inline page stuck at end of title token
            inline_pg = re.search(r'(\d+)\s*$', title)
            if inline_pg and re.match(r'^\d+$', inline_pg.group(1)):
                page  = int(inline_pg.group(1))
                title = title[:inline_pg.start()].strip()
                i = j
            elif j < len(tokens) and re.match(r'^\d+$', tokens[j]):
                page = int(tokens[j])
                i = j + 1
            else:
                i = j
                continue   # no page found, skip

            if title:
                sections.append({'num': num_str, 'title': title,
                                  'page': page,  'chapter': chapter_str})
            continue

        # ── Case C: chapter number token  e.g. "2" ──────────────────────
        m_ch = chapter_num.match(tok)
        if m_ch:
            num_str = tok

            # next non-dot token = title
            j = i + 1
            while j < len(tokens) and dot_line.match(tokens[j]):
                j += 1
            if j >= len(tokens):
                i += 1; continue

            title = re.sub(r'\s*[\.\s]{3,}.*$', '', tokens[j]).strip()
            j += 1

            # skip dots, find page
            while j < len(tokens) and dot_line.match(tokens[j]):
                j += 1

            if j < len(tokens) and re.match(r'^\d+$', tokens[j]):
                page = int(tokens[j])
                i = j + 1
                chapters.append({'num': num_str, 'title': title, 'page': page})
            else:
                i = j
            continue

        # ── Case D: "10 Classical Planning" or "11 Planning..." (num+title fused) ─
        m_fused = re.match(r'^(\d+)\s+(.+)$', tok)
        if m_fused:
            num_str = m_fused.group(1)
            title   = re.sub(r'\s*[\.\s]{3,}.*$', '', m_fused.group(2)).strip()

            j = i + 1
            while j < len(tokens) and dot_line.match(tokens[j]):
                j += 1
            if j < len(tokens) and re.match(r'^\d+$', tokens[j]):
                page = int(tokens[j])
                i = j + 1
                chapters.append({'num': num_str, 'title': title, 'page': page})
            else:
                i = j
            continue

        i += 1

    return chapters, sections


# ── RUN ─────────────────────────────────────────────────────────────────────
chapters, sections = parse_toc_structured(toc_lines)

import re, json

# ── Build toc_json first (if not already done) ──────────────
def toc_lines_to_json(toc_lines):
    dot_line   = re.compile(r'^[.\s]+$')
    roman_skip = re.compile(r'^(Contents|Bibliography|Index|[ivxlcdmIVXLCDM]+)$')
    subsec_pat = re.compile(r'^(\d+)\.(\d+)$')
    chap_pat   = re.compile(r'^(\d+)$')
    fused_pat  = re.compile(r'^(\d+)\s+(.+)$')
    tokens     = [t.strip() for t in toc_lines if t.strip()]
    structure  = {}

    def peek_page(start):
        j = start
        while j < len(tokens):
            if dot_line.match(tokens[j]) or not tokens[j]: j += 1; continue
            if re.match(r'^\d+$', tokens[j]): return int(tokens[j]), j + 1
            break
        return None, start

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if roman_skip.match(tok) or dot_line.match(tok): i += 1; continue

        if subsec_pat.match(tok):
            num = tok; ch = num.split('.')[0]; j = i + 1
            while j < len(tokens) and dot_line.match(tokens[j]): j += 1
            title = ""
            if j < len(tokens) and not re.match(r'^\d+$', tokens[j]):
                title = re.sub(r'\s*[.\s]{3,}.*$', '', tokens[j]).strip(); j += 1
            page, j = peek_page(j)
            if page is None:
                m = re.search(r'\s(\d+)$', title)
                if m: page = int(m.group(1)); title = title[:m.start()].strip()
            if title and page is not None and ch in structure:
                structure[ch]['subtopics'].append({'num': num, 'title': title, 'page': page})
            i = j; continue

        m_fused = fused_pat.match(tok)
        if m_fused and not subsec_pat.match(tok):
            num = m_fused.group(1)
            title = re.sub(r'\s*[.\s]{3,}.*$', '', m_fused.group(2)).strip()
            page, j = peek_page(i + 1)
            if page is not None:
                structure[num] = {'title': title, 'page': page, 'subtopics': []}
            i = j; continue

        if chap_pat.match(tok):
            num = tok; j = i + 1
            while j < len(tokens) and dot_line.match(tokens[j]): j += 1
            if j < len(tokens) and not re.match(r'^\d+$', tokens[j]):
                title = re.sub(r'\s*[.\s]{3,}.*$', '', tokens[j]).strip(); j += 1
            else: i += 1; continue
            page, j = peek_page(j)
            if page is not None:
                structure[num] = {'title': title, 'page': page, 'subtopics': []}
            i = j; continue
        i += 1

    return [{'chapter': n, 'title': d['title'], 'page': d['page'], 'subtopics': d['subtopics']}
            for n, d in sorted(structure.items(), key=lambda x: int(x[0]))]

# Build toc_json NOW so find_page_offset can use it
toc_json = toc_lines_to_json(toc_lines)


# ── Now find_page_offset is safe to call ────────────────────
def find_page_offset(pages, toc_end_pdf_page, first_entry_title, first_book_page=1):
    search_title = first_entry_title.lower().strip()
    for pdf_page in range(toc_end_pdf_page + 1, len(pages) + 1):
        if search_title in pages[pdf_page].lower():
            offset = pdf_page - first_book_page
            return offset
    fallback = toc_end_pdf_page

    return fallback

first_title = toc_json[0]['subtopics'][0]['title']
first_page  = toc_json[0]['subtopics'][0]['page']

offset = find_page_offset(pages, end_page, first_title, first_page)

def extract_subsection_texts(pages, sections, offset):
    results = []
    for idx, sec in enumerate(sections):
        pdf_start = sec['page'] + offset
        pdf_end   = sections[idx + 1]['page'] + offset if idx + 1 < len(sections) else len(pages) + 1

        text = "".join(pages[p] for p in range(pdf_start, pdf_end) if p in pages)

        results.append({
            'num':       sec['num'],
            'title':     sec['title'],
            'chapter':   sec['chapter'],
            'book_page': sec['page'],
            'pdf_page':  pdf_start,
            'text':      text.strip()
        })
    return results

subsections = extract_subsection_texts(pages, sections, offset)

def extract_subsection_texts(pages, sections, offset):
    results = []

    for idx, sec in enumerate(sections):
        pdf_start = sec['page'] + offset

        if idx + 1 < len(sections):
            next_sec  = sections[idx + 1]
            pdf_end   = next_sec['page'] + offset
            # The boundary page belongs to BOTH sections —
            # include it, then cut at the next section's heading
            include_boundary = True
        else:
            pdf_end           = len(pages) + 1
            include_boundary  = False
            next_sec          = None

        # Collect pages — include the boundary page
        page_range = range(pdf_start, pdf_end + 1) if include_boundary else range(pdf_start, pdf_end)
        text = "".join(pages[p] for p in page_range if p in pages)

        # Trim text at the next section's heading on the boundary page
        if include_boundary and next_sec:
            cut_text = _trim_at_next_section(text, next_sec['num'], next_sec['title'])
            text = cut_text

        results.append({
            'num':       sec['num'],
            'title':     sec['title'],
            'chapter':   sec['chapter'],
            'book_page': sec['page'],
            'pdf_page':  pdf_start,
            'text':      text.strip()
        })

    return results


def _trim_at_next_section(text, next_num, next_title):
    """
    Cut text at the point where the next section begins.
    Tries multiple patterns to find the boundary robustly.
    """
    # Pattern 1: "1.2\nTHE FOUNDATIONS..." (number on its own line)
    # Pattern 2: "1.2 The Foundations..."  (number and title on same line)
    # Pattern 3: just the section number alone

    patterns = [
        # number + newline + title (caps or mixed)
        rf'{re.escape(next_num)}\s+{re.escape(next_title[:20])}',
        # number alone on a line
        rf'(?m)^{re.escape(next_num)}\s*$',
        # title alone (fallback)
        rf'{re.escape(next_title[:30])}',
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return text[:m.start()]

    # No boundary found — return full text (safe fallback)
    return text


# ── RUN ─────────────────────────────────────────────────────────────────────
subsections = extract_subsection_texts(pages, sections, offset)

# Verify 1.1 is now complete
s = next(x for x in subsections if x['num'] == '1.1')

def get_section(num):
    s = next((x for x in subsections if x['num'] == num), None)
    if not s:
        print(f"Section '{num}' not found")
        return None
    return s['text']

import re
import json

def toc_lines_to_json(toc_lines):
    dot_line   = re.compile(r'^[.\s]+$')
    roman_skip = re.compile(r'^(Contents|Bibliography|Index|[ivxlcdmIVXLCDM]+)$')
    subsec_pat = re.compile(r'^(\d+)\.(\d+)$')        # "1.1", "18.10"
    chap_pat   = re.compile(r'^(\d+)$')               # "1", "2"
    fused_pat  = re.compile(r'^(\d+)\s+(.+)$')        # "10 Classical Planning"
    inline_pat = re.compile(r'^(.+?)\s+(\d+)\s*$')    # "Title . . . 1003"

    tokens = [t.strip() for t in toc_lines if t.strip()]

    structure = {}   # { '1': { 'title': 'Introduction', 'page': 1, 'subtopics': [...] } }
    current_chapter = None

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # ── skip noise ──────────────────────────────────────────────
        if roman_skip.match(tok) or dot_line.match(tok):
            i += 1; continue

        def peek_page(start):
            """Look ahead past dot lines to find a page number token."""
            j = start
            while j < len(tokens):
                if dot_line.match(tokens[j]) or not tokens[j]:
                    j += 1; continue
                if re.match(r'^\d+$', tokens[j]):
                    return int(tokens[j]), j + 1
                break
            return None, start

        # ── subsection e.g. "1.1" ───────────────────────────────────
        if subsec_pat.match(tok):
            num = tok
            ch  = num.split('.')[0]
            j   = i + 1
            # skip dots to title
            while j < len(tokens) and dot_line.match(tokens[j]): j += 1
            title = ""
            if j < len(tokens) and not re.match(r'^\d+$', tokens[j]):
                title = re.sub(r'\s*[\.\s]{3,}.*$', '', tokens[j]).strip()
                j += 1
            page, j = peek_page(j)
            # inline page stuck at end of title?
            if page is None:
                m = re.search(r'\s(\d+)$', title)
                if m:
                    page  = int(m.group(1))
                    title = title[:m.start()].strip()
            if title and page is not None and ch in structure:
                structure[ch]['subtopics'].append({
                    'num': num, 'title': title, 'page': page
                })
            i = j; continue

        # ── fused chapter+title e.g. "10 Classical Planning" ────────
        m_fused = fused_pat.match(tok)
        if m_fused and not subsec_pat.match(tok):
            num   = m_fused.group(1)
            title = re.sub(r'\s*[\.\s]{3,}.*$', '', m_fused.group(2)).strip()
            page, j = peek_page(i + 1)
            if page is not None:
                structure[num] = {'title': title, 'page': page, 'subtopics': []}
                current_chapter = num
            i = j; continue

        # ── bare chapter number e.g. "2" ─────────────────────────────
        if chap_pat.match(tok):
            num = tok
            j   = i + 1
            while j < len(tokens) and dot_line.match(tokens[j]): j += 1
            if j < len(tokens) and not re.match(r'^\d+$', tokens[j]):
                title = re.sub(r'\s*[\.\s]{3,}.*$', '', tokens[j]).strip()
                j += 1
            else:
                i += 1; continue
            page, j = peek_page(j)
            if page is not None:
                structure[num] = {'title': title, 'page': page, 'subtopics': []}
                current_chapter = num
            i = j; continue

        i += 1

    # ── convert to clean list ────────────────────────────────────────
    result = []
    for num, data in sorted(structure.items(), key=lambda x: int(x[0])):
        result.append({
            'chapter': num,
            'title':   data['title'],
            'page':    data['page'],
            'subtopics': data['subtopics']
        })

    return result


# ── RUN ─────────────────────────────────────────────────────────────────────
toc_json = toc_lines_to_json(toc_lines)

# Save to file
with open('toc.json', 'w', encoding='utf-8') as f:
    json.dump(toc_json, f, indent=2, ensure_ascii=False)

import os

os.makedirs('sections', exist_ok=True)

for chapter in toc_json:
    ch_num   = chapter['chapter']
    ch_title = chapter['title']

    for sub in chapter['subtopics']:
        num   = sub['num']
        title = sub['title']

        text = get_section(num)

        if not text:
            print(f"  ⚠ No text found for {num}")
            continue

        # save to file
        safe_title = title[:30].replace(' ', '_').replace('/', '_')
        fname = f"sections/{num}_{safe_title}.txt"
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(f"[{num}] {title}\n")
            f.write(f"Chapter {ch_num}: {ch_title}\n")
            f.write("=" * 60 + "\n\n")
            f.write(text)

# Add text into toc_json itself
for chapter in toc_json:
    for sub in chapter['subtopics']:
        sub['text'] = get_section(sub['num']) or ""

# Save the enriched JSON
with open('toc_with_text.json', 'w', encoding='utf-8') as f:
    json.dump(toc_json, f, indent=2, ensure_ascii=False)

# ====== GET SUBTOPIC PAGES FOR A TOPIC ======

def get_subtopic_pages(topic_number):
    """
    Given a topic (chapter) number, returns a list of JSON objects
    with each subtopic's num, title, start page, and end page.

    Args:
        topic_number (int or str): e.g. 1, 5, "10"

    Returns:
        list of dicts: [{'num': '1.1', 'title': '...', 'start': 1, 'end': 5}, ...]
    """
    topic_number = str(topic_number)

    # Find the matching chapter
    chapter = next((c for c in toc_json if c['chapter'] == topic_number), None)

    if chapter is None:
        print(f"Topic {topic_number} not found. "
              f"Available topics: {[c['chapter'] for c in toc_json]}")
        return []

    subtopics = chapter['subtopics']

    if not subtopics:
        print(f"Topic {topic_number} has no subtopics.")
        return []

    result = []

    for idx, sub in enumerate(subtopics):
        start = sub['page']

        # end = page before next subtopic starts, or chapter boundary
        if idx + 1 < len(subtopics):
            end = subtopics[idx + 1]['page'] - 1
        else:
            # Last subtopic — ends at the next chapter's start, or EOF
            next_chapter = next(
                (c for c in toc_json if int(c['chapter']) == int(topic_number) + 1),
                None
            )
            end = (next_chapter['page'] - 1) if next_chapter else len(pages) - offset

        result.append({
            'num':   sub['num'],
            'title': sub['title'],
            'start': start,
            'end':   end
        })

    return result


# ====== LOOP OVER ALL TOPICS ======

num_topics = len(toc_json)


for chapter in toc_json:
    topic_num = chapter['chapter']
    print(f"── Chapter {topic_num}: {chapter['title']} ──")

    subtopic_pages = get_subtopic_pages(topic_num)

    for s in subtopic_pages:
        print(f"  {s['num']:>6}  pp. {s['start']:>4} – {s['end']:>4}  |  {s['title']}")

    print()






# %%
toc_json[0]

# %%
def flatten_toc(toc_json):
    flat_list = []
    
    for chapter in toc_json:
        for sub in chapter.get("subtopics", []):
            flat_list.append({
                "num": sub.get("num"),
                "title": sub.get("title"),
                "text": sub.get("text")
            })
    
    return flat_list

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def match_lecture_to_toc_with_context(lecture_title, toc_json, top_k=3, window=2):
    
    # --- Flatten ---
    flat_toc = flatten_toc(toc_json)
    
    titles = [item["title"] for item in flat_toc]
    
    # --- Encode ---
    lecture_emb = model.encode(lecture_title, convert_to_tensor=True)
    title_embs = model.encode(titles, convert_to_tensor=True)
    
    # --- Similarity ---
    scores = util.cos_sim(lecture_emb, title_embs)[0]
    
    # --- Top K matches ---
    top_indices = scores.topk(k=top_k).indices.tolist()
    
    # --- Collect context indices ---
    selected_indices = set()
    
    for idx in top_indices:
        for offset in range(-window, window + 1):
            new_idx = idx + offset
            
            if 0 <= new_idx < len(flat_toc):
                selected_indices.add(new_idx)
    
    # --- Sort indices (important for order) ---
    selected_indices = sorted(selected_indices)
    
    # --- Build results ---
    results = []
    
    for idx in selected_indices:
        item = flat_toc[idx]
        
        results.append({
            "num": item["num"],
            "title": item["title"],
            "text": item["text"]
        })
    
    return results

# %%
import fitz  # PyMuPDF
import re
import os

import fitz
import re

def normalize_text(text):
    text = text.replace("\r", "")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[ \t]+", " ", text)  # normalize spaces
    return text


def extract_lecture_info(pdf_path):
    doc = fitz.open(pdf_path)
    text = doc[0].get_text()



    text = normalize_text(text)

    print(text)

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    lecture_num = None
    lecture_title = None

    for i, line in enumerate(lines):

        # flexible lecture match
        m = re.search(r"Lecture\s*-\s*(\d+)", line, re.IGNORECASE)

        if m:
            lecture_num = int(m.group(1))

            # find title: next meaningful line (skip metadata noise)
            for j in range(i + 1, min(i + 5, len(lines))):
                candidate = lines[j]

                # filter out noise lines
                if any(x in candidate.lower() for x in [
                    "department",
                    "institute",
                    "university",
                    "prof",
                    "india",
                    "lecture"
                ]):
                    continue

                lecture_title = candidate
                return lecture_num, lecture_title

    return None, None

import re
import fitz

def extract_lecture_info(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = doc[0].get_text()

        text = normalize_text(text)

        lines = [l.strip() for l in text.split("\n") if l.strip()]

        lecture_num = None
        lecture_title = None

        # =========================
        # PRIMARY LOGIC (your original)
        # =========================
        for i, line in enumerate(lines):

            m = re.search(r"Lecture\s*-\s*(\d+)", line, re.IGNORECASE)

            if m:
                lecture_num = int(m.group(1))

                for j in range(i + 1, min(i + 5, len(lines))):
                    candidate = lines[j]

                    if any(x in candidate.lower() for x in [
                        "department",
                        "institute",
                        "university",
                        "prof",
                        "india",
                        "lecture"
                    ]):
                        continue

                    lecture_title = candidate
                    return lecture_num, lecture_title

        # =========================
        # FALLBACK LOGIC (NO HASH / RELAXED)
        # =========================
        for i, line in enumerate(lines):


            # look for ANY line containing "Lecture"
            if "lecture" in line.lower():

                # extract number manually (no strict pattern)
                nums = re.findall(r"\d+", line)

                if nums:
                    lecture_num = int(nums[0])

                    # title search same as before
                    for j in range(i + 1, min(i + 5, len(lines))):
                        candidate = lines[j]

                        if any(x in candidate.lower() for x in [
                            "department",
                            "institute",
                            "university",
                            "prof",
                            "india",
                            "lecture"
                        ]):
                            continue

                        lecture_title = candidate
                        return lecture_num, lecture_title
        # =========================
        # FINAL FAILURE LOG
        # =========================
        print("\n[FAILURE] Could not extract lecture info from:")
        print("FILE:", pdf_path)
        print(text)
        print("FIRST 15 LINES:")
        for l in lines[:15]:
            print("  ", l)
        return None, None

    except Exception as e:
        print("ERROR extracting lecture info:", str(e))
        return None, None
    


def get_pdf_by_lecture(folder_path, lecture_number):
    target_file = f"lec{lecture_number}.pdf"
    full_path = os.path.join(folder_path, target_file)
    
    if os.path.exists(full_path):
        return full_path
    return None



failed_indices = []
folder = "/home/roy.2/rehtorical_segmentation_ai_Course/lecture_pdf"

matched_subtopic_name_lst=[]
for i in range(len(df)):
   
    print(i)

    lec_num = df.iloc[i]['lecture_number']
    print("lec_num",lec_num)

    #read_pdf
    pdf_path = get_pdf_by_lecture(folder, lec_num)


    if pdf_path:
        lec_num, lec_title = extract_lecture_info(pdf_path)

    else:
        print("PDF not found")
        failed_indices.append(i)
        continue

    ###matching 


    matches = match_lecture_to_toc_with_context(lec_title, toc_json)

    matches_lst_json = []

    for m in matches:
        matches_lst_json.append({
            "num": m["num"],
            "title": m["title"],
            "text": m["text"]
        })

    matched_subtopic_name_lst.append(matches_lst_json)

    

    


print("Failed indices:", failed_indices)

df = df.drop(index=failed_indices).reset_index(drop=True)

df['matched_subtopics']=matched_subtopic_name_lst
    

    

# %%
df

# %% [markdown]
# helper functions 

# %%
import re
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# =========================
# HELPER: CLEAN TEXT
# =========================
def clean_text(text):
    text = re.sub(r'\n+', ' ', str(text))         # remove line breaks
    text = re.sub(r'\s+', ' ', text)              # normalize spaces
    text = re.sub(r'\b\d+\b', '', text)           # remove standalone numbers
    return text.strip()

# =========================
# HELPER: SENTENCE SPLIT (IMPROVED)
# =========================
def split_sentences(text):
    sents = sent_tokenize(text)
    
    refined = []
    for s in sents:
        # split further on semicolons (important for lecture/book text)
        parts = re.split(r';|\.\s+(?=[A-Z])', s)
        refined.extend([p.strip() for p in parts if p.strip()])
    
    return refined



IGNORE_TITLES = {
    "summary",
    "bibliographical and historical notes",
    "exercises",
    'Summary, Bibliographical and Historical Notes, Exercises' 
}
IGNORE_TITLES = [
    "summary",
    "bibliographical and historical notes",
    "exercises"
]

def is_ignored(title):
    title = title.lower()
    return any(x in title for x in IGNORE_TITLES)

def create_chunks(toc_json, k=6):
    all_chunks = []
    
    for chapter in toc_json:
        chapter_num = chapter.get('chapter')
        
        for sub in chapter.get('subtopics', []):
            title = sub.get('title', "")

            # =========================
            # SKIP IRRELEVANT SECTIONS
            # =========================
            if is_ignored(title):
                continue

            sub_num = sub.get('num')
            text = sub.get('text', "")

            text = clean_text(text)
            sentences = split_sentences(text)

            for i in range(0, len(sentences), k):
                chunk_sents = sentences[i:i+k]
                chunk_text = " ".join(chunk_sents)

                all_chunks.append({
                    "chapter": chapter_num,
                    "subtopic_num": sub_num,
                    "title": title,
                    "chunk": chunk_text
                })

    
    return all_chunks

# =========================
# RUN
# =========================



# %%
import re
from nltk.tokenize import sent_tokenize

def parse_ocr_text(text):
    results = []

    # pattern to extract <tag>content</tag>
    pattern = re.findall(r"<(.*?)>(.*?)</\1>", str(text), re.DOTALL)

    for tag, content in pattern:
        tag = tag.strip()
        content = content.strip()

        # ❌ skip title
        if tag.lower() == "title":
            continue

        # ❌ skip diagram if contains NPTEL
        if tag.lower() == "diagram" and "nptel" in content.lower():
            continue

        # split into sentences
        sentences = sent_tokenize(content)

        for sent in sentences:
            results.append({
                "tag": tag,
                "text": sent.strip()
            })

    return results

import ast
from nltk.tokenize import sent_tokenize

def flatten_segments(segment_info):
    if isinstance(segment_info, str):
        segment_info = ast.literal_eval(segment_info)

    results = []

    for seg in segment_info:
        label = seg.get("label")
        text = seg.get("text", "")

        sentences = sent_tokenize(text)

        for sent in sentences:
            results.append({
                "segment": label,
                "text": sent.strip()
            })

    return results

# =========================
# INSTALL (if needed)
# =========================
# !pip install sentence-transformers faiss-cpu

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re


# =========================
# LOAD MODEL
# =========================
similarity_model = SentenceTransformer('/home/roy.2/models/all-MiniLM-L6-v2')

# =========================
# 1. PREP TEXT
# =========================
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return text.split()


from nltk.tokenize import sent_tokenize

def get_ocr_sentences(parsed_ocr):
    sentences = []
    
    for item in parsed_ocr:
        text = item.get("text", "")
        
        # split into sentences
        sents = sent_tokenize(str(text))
        
        # add to final list
        sentences.extend(sents)
    
    return sentences

from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_sentences(sentences):
    seen = set()
    cleaned = []

    for s in sentences:
        s = s.strip()

        # skip empty
        if not s:
            continue

        # count words
        words = re.findall(r'\b\w+\b', s)

        # remove one-word sentences
        if len(words) <= 1:
            continue

        # remove duplicates (case-insensitive)
        key = s.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(s)

    return cleaned

def build_similarity_matrix(
    ocr_sentences,
    transcript_sentence_nested,
    all_segments_sentences,
    similarity_model="/home/roy.2/models/all-MiniLM-L6-v2"
):
    """
    Rows    -> OCR + Book sentences
    Columns -> Transcript sentences
    """

    # =========================
    # LOAD MODEL IF STRING
    # =========================
    if isinstance(similarity_model, str):
        similarity_model = SentenceTransformer(similarity_model)

    # =========================
    # FLATTEN OCR
    # =========================
    ocr_flat = [s for x in ocr_sentences for s in (x if isinstance(x, list) else [x])]

    # =========================
    # FLATTEN TRANSCRIPT
    # =========================
    transcript_flat = [
        s for x in transcript_sentence_nested for s in (x if isinstance(x, list) else [x])
    ]

    # =========================
    # FLATTEN BOOK
    # =========================
    book_flat = [
        s for seg in all_segments_sentences
        for chunk in seg
        for s in (chunk if isinstance(chunk, list) else [chunk])
    ]

    book_flat = clean_sentences(book_flat)

    print(len(ocr_flat),len(transcript_flat), len(book_flat))
    # =========================
    # ROWS + COLS
    # =========================
    row_texts = ocr_flat + book_flat
    col_texts = transcript_flat

    # =========================
    # EMBEDDINGS
    # =========================
    # =========================
    # 1. SEMANTIC SIMILARITY
    # =========================
    row_emb = similarity_model.encode(row_texts, normalize_embeddings=True)
    col_emb = similarity_model.encode(col_texts, normalize_embeddings=True)

    semantic_sim = cosine_similarity(row_emb, col_emb)

    # =========================
    # 2. LEXICAL SIMILARITY (TF-IDF)
    # =========================
    # tfidf = TfidfVectorizer(stop_words="english")

    # all_texts = row_texts + col_texts
    # tfidf_matrix = tfidf.fit_transform(all_texts)

    # row_tfidf = tfidf_matrix[:len(row_texts)]
    # col_tfidf = tfidf_matrix[len(row_texts):]

    # lexical_sim = cosine_similarity(row_tfidf, col_tfidf)
    # alpha=0.5

    # sim_matrix = alpha * semantic_sim + (1 - alpha) * lexical_sim 
    sim_matrix= semantic_sim

    return sim_matrix, row_texts, col_texts



def split_sentences(text):
    sents = sent_tokenize(text)
    refined = []
    for s in sents:
        # splits further on semicolons and period+Capital letter
        parts = re.split(r';|\.\s+(?=[A-Z])', s)
        refined.extend([p.strip() for p in parts if p.strip()])
    return refined

from nltk.tokenize import sent_tokenize



def build_segment_sentence_structure(results_per_segment):
    """
    Returns:
    [
        [sent1+sent2, sent2+sent3, ...],   # segment 1 (overlapping windows)
        [sent1+sent2, ...],                # segment 2
        ...
    ]
    """

    all_segments_sentences = []

    for seg in results_per_segment:
        segment_sentences = []

        for r in seg['results']:
            chunk_text = str(r['text'])

            # split into sentences
            sentences = sent_tokenize(chunk_text)

            # create overlapping 2-sentence windows
            for i in range(len(sentences) - 1):
                combined = sentences[i] + " " + sentences[i + 1]
                segment_sentences.append(combined)

        all_segments_sentences.append(segment_sentences)

    return all_segments_sentences

    # all_segments_sentences = []

    # for seg in results_per_segment:
    #     segment_sentences = []

    #     for r in seg['results']:
    #         chunk_text = str(r['chunk'])

    #         # split into sentences
    #         sentences = sent_tokenize(chunk_text)

    #         # FLATTEN (extend instead of append)
    #         segment_sentences.extend(sentences)

    #     all_segments_sentences.append(segment_sentences)

    # return all_segments_sentences

def flatten_sentences(nested_list):
    flat = []
    
    for block in nested_list:
        for sentence in block:
            flat.append(sentence)
    
    return flat

import re






# %%
import re
from typing import List, Dict

def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter (works reasonably for lecture-style text).
    """
    # split on ., ?, ! followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # remove empty strings
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentences(sentences: List[str], chunk_size: int = 5) -> List[str]:
    """
    Group sentences into chunks of fixed size.
    """
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def create_chunks(nested_data: List[Dict], chunk_size: int = 5) -> List[str]:
    """
    1. Loop over nested structure
    2. Extract text
    3. Split into sentences
    4. Chunk into groups of `chunk_size`
    """
    all_sentences = []

    for item in nested_data:
        text = item.get("text", "")
        if not text:
            continue

        sentences = split_into_sentences(text)
        all_sentences.extend(sentences)

    return chunk_sentences(all_sentences, chunk_size)

import numpy as np
from sentence_transformers import CrossEncoder

def build_paraphrase_matrix(
    row_texts,
    col_texts,
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
):
    """
    Builds paraphrase similarity matrix using CrossEncoder
    and returns analyzed results similar to analyze_sim_matrix().
    """

    # =========================
    # LOAD MODEL
    # =========================
    model = CrossEncoder(model_name)

    # =========================
    # PAIR CREATION
    # =========================
    pairs = [(r, c) for r in row_texts for c in col_texts]

    # =========================
    # PREDICT SCORES
    # =========================
    scores = model.predict(pairs)

    para_matrix = scores.reshape(len(row_texts), len(col_texts))

    # =========================
    # NORMALIZATION
    # =========================
    min_val = para_matrix.min()
    max_val = para_matrix.max()

    if max_val - min_val != 0:
        para_matrix = (para_matrix - min_val) / (max_val - min_val)
    else:
        para_matrix = np.zeros_like(para_matrix)

    # =========================
    # THRESHOLDING
    # =========================
    THRESHOLD = 0.65
    top_k = 5

    para_thresholded = np.where(para_matrix >= THRESHOLD, para_matrix, 0.0)

    # =========================
    # TOP MATCH EXTRACTION
    # =========================
    top_matches_per_col = []

    for j in range(para_thresholded.shape[1]):

        col_scores = para_thresholded[:, j]

        # top-k indices
        if top_k < len(col_scores):
            top_indices = np.argpartition(col_scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(col_scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(col_scores)[::-1]

        matches = []

        for i in top_indices:
            score = col_scores[i]

            if score > 0:
                matches.append({
                    "row_index": int(i),
                    "row_text": row_texts[i],
                    "score": float(score)
                })

        top_matches_per_col.append({
            "col_index": int(j),
            "col_text": col_texts[j],
            "matches": matches,
            "nonzero_count": int(np.count_nonzero(col_scores))
        })

    return para_matrix, para_thresholded, top_matches_per_col

# %%
import numpy as np

def inspect_similarity_matrix(
    sim_matrix,
    row_texts,
    col_texts,
    threshold=0.7,
    top_k=5,
    normalize=True
):
    """
    Takes similarity matrix and returns:
    - thresholded matrix
    - column-wise match stats
    - top-k matches per column
    """

    # =========================
    # NORMALIZATION
    # =========================
    if normalize:
        sim = sim_matrix
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    else:
        sim = sim_matrix

    # =========================
    # THRESHOLDING
    # =========================
    sim_thresholded = np.where(sim >= threshold, sim, 0.0)

    col_nonzero_counts = np.count_nonzero(sim_thresholded, axis=0)

    # =========================
    # INSPECTION
    # =========================
    top_matches_per_col = []

    for j in range(sim.shape[1]):  # columns = transcript

        col_scores = sim[:, j]

        top_indices = np.argsort(col_scores)[::-1][:top_k]

        matches = []

        for i in top_indices:
            score = col_scores[i]

            if score >= threshold:
                matches.append({
                    "row_index": int(i),
                    "row_text": row_texts[i],
                    "score": float(score)
                })

        top_matches_per_col.append({
            "col_index": j,
            "col_text": col_texts[j],
            "matches": matches,
            "nonzero_count": int(col_nonzero_counts[j])
        })

    return sim, sim_thresholded, col_nonzero_counts, top_matches_per_col

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def build_sim_matrix(
    row_texts,
    col_emb,
    similarity_model
):
    """
    Encodes row_texts and computes cosine similarity with col_emb.

    Returns:
        sim_matrix
        row_embeddings
    """

    # =========================
    # ENCODE ROWS
    # =========================
    row_emb = similarity_model.encode(
        row_texts,
        normalize_embeddings=True
    )

    row_emb = np.array(row_emb).astype("float32")

    # =========================
    # COSINE SIMILARITY
    # =========================
    sim_matrix = cosine_similarity(row_emb, col_emb)

    return sim_matrix, row_emb

def analyze_sim_matrix(
    sim_matrix,
    row_texts,
    col_texts,
    threshold=0.7,
    top_k=5,
    normalize=True
):
    # =========================
    # NORMALIZE
    # =========================
    sim = sim_matrix
    if normalize:
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)

    # =========================
    # THRESHOLD
    # =========================
    sim_thresh = np.where(sim >= threshold, sim, 0.0)
    col_nonzero_counts = np.count_nonzero(sim_thresh, axis=0)

    # =========================
    # TOP MATCHES
    # =========================
    top_matches_per_col = []

    for j in range(sim.shape[1]):
        col_scores = sim[:, j]
        top_indices = np.argsort(col_scores)[::-1][:top_k]

        matches = []
        for i in top_indices:
            if col_scores[i] >= threshold:
                matches.append({
                    "row_index": int(i),
                    "row_text": row_texts[i],
                    "score": float(col_scores[i])
                })

        top_matches_per_col.append({
            "col_index": j,
            "col_text": col_texts[j],
            "matches": matches,
            "nonzero_count": int(col_nonzero_counts[j])
        })

    return sim, sim_thresh, col_nonzero_counts, top_matches_per_col

# %%
import numpy as np
from sentence_transformers import CrossEncoder

def build_paraphrase_matrix(
    row_texts,
    col_texts,
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
):
    """
    Builds paraphrase similarity matrix using CrossEncoder
    and returns analyzed results similar to analyze_sim_matrix().
    """

    # =========================
    # LOAD MODEL
    # =========================
    model = CrossEncoder(model_name)

    # =========================
    # PAIR CREATION
    # =========================
    pairs = [(r, c) for r in row_texts for c in col_texts]

    # =========================
    # PREDICT SCORES
    # =========================
    scores = model.predict(pairs)

    para_matrix = scores.reshape(len(row_texts), len(col_texts))

    # =========================
    # NORMALIZATION
    # =========================
    min_val = para_matrix.min()
    max_val = para_matrix.max()

    if max_val - min_val != 0:
        para_matrix = (para_matrix - min_val) / (max_val - min_val)
    else:
        para_matrix = np.zeros_like(para_matrix)

    # =========================
    # THRESHOLDING
    # =========================
    THRESHOLD = 0.70
    top_k = 5

    para_thresholded = np.where(para_matrix >= THRESHOLD, para_matrix, 0.0)

    # =========================
    # TOP MATCH EXTRACTION
    # =========================
    top_matches_per_col = []

    for j in range(para_thresholded.shape[1]):

        col_scores = para_thresholded[:, j]

        # top-k indices
        if top_k < len(col_scores):
            top_indices = np.argpartition(col_scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(col_scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(col_scores)[::-1]

        matches = []

        for i in top_indices:
            score = col_scores[i]

            if score > 0:
                matches.append({
                    "row_index": int(i),
                    "row_text": row_texts[i],
                    "score": float(score)
                })

        top_matches_per_col.append({
            "col_index": int(j),
            "col_text": col_texts[j],
            "matches": matches,
            "nonzero_count": int(np.count_nonzero(col_scores))
        })

    return para_matrix, para_thresholded, top_matches_per_col

# %%
slide_text= df.iloc[150]['ocr_text']
slide_text

# %%
df.columns

# %%
df.iloc[150]['lecture_number']

# %%
def build_final_alignment_json(
    col_texts,
    top_matches_per_col_ocr,
    book_top_matches,
    top_para_matches_per_col
):
    """
    Builds final explainable alignment JSON combining:
    OCR + Book + Paraphrase matches
    """

    final_alignment = []

    for j in range(len(col_texts)):

        entry = {
            "transcript_sentence_index": j,
            "transcript_sentence_text": col_texts[j],

            "ocr_matches": [],
            "book_matches": [],
            "paraphrase_matches_from_book": []
        }

        # =========================
        # OCR MATCHES
        # =========================
        if j < len(top_matches_per_col_ocr):
            entry["ocr_matches"] = top_matches_per_col_ocr[j].get("matches", [])

        # =========================
        # BOOK MATCHES
        # =========================
        if j < len(book_top_matches):
            entry["book_matches"] = book_top_matches[j].get("matches", [])

        # =========================
        # PARAPHRASE MATCHES
        # =========================
        if j < len(top_para_matches_per_col):
            entry["paraphrase_matches_from_book"] = top_para_matches_per_col[j].get("matches", [])

        # =========================
        # COMBINED SUPPORT
        # =========================
        combined = (
            entry["ocr_matches"] +
            entry["book_matches"] +
            entry["paraphrase_matches_from_book"]
        )

        combined = sorted(combined, key=lambda x: x.get("score", 0), reverse=True)
        entry["final_top_3_support"] = combined[:3]

        # =========================
        # AGREEMENT SCORE
        # =========================
        ocr_score = max([m.get("score", 0) for m in entry["ocr_matches"]] or [0])
        book_score = max([m.get("score", 0) for m in entry["book_matches"]] or [0])
        para_score = max([m.get("score", 0) for m in entry["paraphrase_matches_from_book"]] or [0])

        entry["agreement_score"] = float(
            0.2 * ocr_score +
            0.7 * book_score +
            0.1* para_score
        )

        # =========================
        # SUPPORT SUMMARY
        # =========================
        entry["support_summary"] = {
            "ocr_count": len(entry["ocr_matches"]),
            "book_count": len(entry["book_matches"]),
            "paraphrase_count_from_book": len(entry["paraphrase_matches_from_book"])
        }

        final_alignment.append(entry)

    return final_alignment

# %%

final_alignment_lst=[]
for i in range(len(df)):

    print(i)

    slide_text= df.iloc[i]['ocr_text']

    parsed_ocr = parse_ocr_text(slide_text)


    segment_info = df.iloc[i]['segment_info']
    
    flat_output_transcript = flatten_segments(segment_info)


    matched_subtopics_for_this= df.iloc[i]['matched_subtopics']

    book_chunks = create_chunks(matched_subtopics_for_this, chunk_size=5)


    chunk_texts = book_chunks

    # =========================
    # 3. BM25 (KEYWORD SEARCH)
    # =========================
    tokenized_corpus = [tokenize(t) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    chunk_embeddings = similarity_model.encode(chunk_texts)
    chunk_embeddings = np.array(chunk_embeddings).astype('float32')

    # normalize for cosine similarity
    faiss.normalize_L2(chunk_embeddings)

    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(chunk_embeddings)


    #query 


    results_per_segment = []

    for seg in segment_info:

        query_text = seg['text']

        # =========================
        # BM25 SCORES
        # =========================
        tokenized_query = tokenize(query_text)
        bm25_scores = bm25.get_scores(tokenized_query)

        bm25_scores = np.array(bm25_scores)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)

        # =========================
        # DENSE (FAISS)
        # =========================
        query_emb = similarity_model.encode([query_text]).astype('float32')
        faiss.normalize_L2(query_emb)

        k = 20
        distances, indices = index.search(query_emb, k)

        dense_scores = np.zeros(len(chunk_texts))
        for idx, score in zip(indices[0], distances[0]):
            dense_scores[idx] = score

        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)

        # =========================
        # HYBRID SCORE
        # =========================
        alpha = 0.6
        beta = 0.4

        hybrid_scores = alpha * dense_scores + beta * bm25_scores

        # =========================
        # TOP K
        # =========================
        top_k = 5
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        # top_chunks = []
        # for idx in top_indices:
        #     item = book_chunks[idx].copy()
        #     item['segment_id'] = seg['segment_id']
        #     item['segment_label'] = seg['label']
        #     item['query'] = query_text
        #     item['hybrid_score'] = float(hybrid_scores[idx])
        #     top_chunks.append(item)
        top_chunks = []
        for idx in top_indices:
            item = {"text": book_chunks[idx]}   # FIX HERE

            item['segment_id'] = seg['segment_id']
            item['segment_label'] = seg['label']
            item['query'] = query_text
            item['hybrid_score'] = float(hybrid_scores[idx])

            top_chunks.append(item)

        results_per_segment.append({
            "segment_id": seg['segment_id'],
            "label": seg['label'],
            "results": top_chunks
        })
   


    ##############sentence similarity matrix ##############

    # slide sentences 
    parsed_ocr = parse_ocr_text(slide_text)

    ocr_sentences = get_ocr_sentences(parsed_ocr)

    all_segments_sentences = build_segment_sentence_structure(results_per_segment)
    print(len(all_segments_sentences))
    # all_segments_sentences = flatten_sentences(all_segments_sentences)
    # print(len(all_segments_sentences))


    transcript_sentence_nested = []

    for item in flat_output_transcript:
        text = item['text']
        
        # split into sentences
        sentences = sent_tokenize(str(text))
        
        transcript_sentence_nested.append(sentences)
    transcript_sentence_nested = flatten_sentences(transcript_sentence_nested)
    print("transcript_sentence_nested",len(transcript_sentence_nested))

    #############################dslide ocr similarity #################################################
    #similarity matrix 

    ocr_row_texts = ocr_sentences
    book_row_texts = all_segments_sentences
    col_texts = transcript_sentence_nested

    ocr_emb = similarity_model.encode(ocr_row_texts, normalize_embeddings=True)
    col_emb = similarity_model.encode(col_texts, normalize_embeddings=True)

    ocr_sim_matrix = cosine_similarity(ocr_emb, col_emb)

    sim_matrix_ocr, sim_thresholded, col_counts, top_matches_per_col_ocr = inspect_similarity_matrix(
    ocr_sim_matrix,
    ocr_row_texts,
    col_texts,
    threshold=0.7,
    top_k=5,
    normalize=True
    )

    #############################book similarity #################################################
    book_flat = []

    for seg in all_segments_sentences:
        for chunk in seg:
            # if chunk is a list of sentences
            if isinstance(chunk, list):
                for sentence in chunk:
                    book_flat.append(sentence)
            else:
                book_flat.append(chunk)

    # clean empty strings
    book_flat = [s for s in book_flat if isinstance(s, str) and len(s.strip()) > 0]
    book_row_texts=book_flat

    # remove duplicates while preserving order
    seen = set()
    book_flat_clean = []

    for s in book_flat:
        if not isinstance(s, str):
            continue

        s = s.strip()

        # =========================
        # remove empty / single-word sentences
        # =========================
        if len(s.split()) <= 1:
            continue

        # =========================
        # remove duplicates
        # =========================
        if s not in seen:
            seen.add(s)
            book_flat_clean.append(s)

    book_row_texts = book_flat_clean

    book_emb = similarity_model.encode(book_row_texts, normalize_embeddings=True)

    book_sim_matrix = cosine_similarity(book_emb, col_emb)

    book_sim, book_thresh, book_counts, book_top_matches = analyze_sim_matrix(
    book_sim_matrix,
    book_row_texts,
    col_texts,
    threshold=0.7,
    top_k=5
    )

    ########################paraphrase similarity btwn book and lecture###############

    para_matrix, para_thresh, top_para_matches_per_col = build_paraphrase_matrix(
    row_texts=book_row_texts,
    col_texts=col_texts
    )

    ##########################3alignment######################################
   
    final_alignment = build_final_alignment_json(
        col_texts=col_texts,
        top_matches_per_col_ocr=top_matches_per_col_ocr,
        book_top_matches=book_top_matches,
        top_para_matches_per_col=top_para_matches_per_col
        )
    
    final_alignment_lst.append(final_alignment)
    


# %%
df['final_alignment_lst']=final_alignment_lst

df.to_csv('/home/roy.2/rehtorical_segmentation_ai_Course/ai_df_lecture_transcript/source_mapped_segment_all_lectures.csv',index=False)


