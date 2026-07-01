# -*- coding: utf-8 -*-
"""
NPTEL Lecture Transcripts Downloader
=====================================
1. Scrapes all CS courses from nptel.ac.in/courses
2. Filters NOC Computer Science courses
3. Downloads English-Verified transcripts for each course

Fixes included:
  - Google Drive viewer URLs are converted to direct download URLs
  - HTML-response guard (catches auth-wall redirects)
  - Retry logic on failed downloads
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import os, re, sys, time, logging, requests
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_SAVE_DIR   = Path("/home/roy.2/rehtorical_segmentation_ai_Course")
DOWNLOAD_FOLDER = "nptel_lecture_notes"
MAX_RETRIES     = 3
LANGUAGE        = "English-Verified"
HEADERS         = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
log_path = os.path.join(DOWNLOAD_FOLDER, "download_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info("=" * 60)
log.info("NPTEL Downloader -- " + datetime.now().strftime("%Y-%m-%d %H:%M"))
log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — SCRAPE COURSE LIST
# ══════════════════════════════════════════════════════════════════════════════

def scrape_courses():
    BASE_URL = "https://nptel.ac.in"
    URL      = "https://nptel.ac.in/courses"

    log.info("Downloading course list from NPTEL...")
    html = requests.get(URL, headers=HEADERS, timeout=60).text
    log.info(f"  HTML size: {len(html)}")

    soup    = BeautifulSoup(html, "lxml")
    results = []
    seen    = set()

    for div in soup.find_all("div"):
        text  = div.get_text("\n", strip=True)
        lines = [x.strip() for x in text.split("\n") if x.strip()]

        if len(lines) < 4:
            continue

        title      = lines[0]
        discipline = lines[1]
        professor  = lines[2]
        institute  = lines[3]

        if (
            len(title) < 5 or len(title) > 200
            or "Search" in title or "Select" in title or "Login" in title
            or len(discipline) > 80
            or ("IIT" not in institute and "IIIT" not in institute)
        ):
            continue

        # Find course URL
        course_url = None
        for a in div.find_all("a", href=True):
            href = a["href"]
            if re.search(r"/courses/\d+", href):
                course_url = urljoin(BASE_URL, href)
                break
        if not course_url:
            m = re.search(r"https://nptel\.ac\.in/courses/\d+", str(div))
            if m:
                course_url = m.group(0)

        key = (title, discipline, professor, institute)
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "course_title": title,
            "discipline":   discipline,
            "professor":    professor,
            "institute":    institute,
            "course_url":   course_url,
        })

    df = pd.DataFrame(results)
    log.info(f"  Total courses found: {len(df)}")
    return df


def filter_noc_cs(df):
    df = df[df["discipline"].str.contains("Computer Science", case=False, na=False)]
    df = df[df["course_title"].str.contains("NOC", case=False, na=False)]
    df = df[df["course_title"] != "NOC:An Introduction to Artificial Intelligence"]
    df = df.reset_index(drop=True)
    log.info(f"  NOC CS courses after filter: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — DOWNLOAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_pdf_url(url: str) -> str:
    """
    Converts any Google Drive viewer / sharing URL to a direct download URL.
    Non-Drive URLs are returned unchanged.

    Handles:
      https://drive.google.com/file/d/<ID>/view?...
      https://drive.google.com/open?id=<ID>
      https://drive.google.com/uc?id=<ID>
    """
    if not url:
        return url

    # /file/d/<ID>/ pattern (viewer or edit link)
    m = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return f"https://drive.usercontent.google.com/uc?id={m.group(1)}&export=download"

    # ?id=<ID> pattern (open / uc links)
    m = re.search(r"drive\.google\.com/.*[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return f"https://drive.usercontent.google.com/uc?id={m.group(1)}&export=download"

    return url


def download_file(session: requests.Session, url: str, dest_path: Path) -> bool:
    """
    Download a file from url to dest_path.
    Automatically resolves Google Drive viewer URLs to direct download links.
    Returns True on success, False on failure.
    """
    direct_url = resolve_pdf_url(url)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"  Attempt {attempt}: {direct_url}")
            resp = session.get(direct_url, stream=True, timeout=60)
            resp.raise_for_status()

            # Guard: reject HTML pages (auth-wall redirects, viewer pages)
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                log.error(
                    f"  Got HTML instead of PDF — file may require authentication.\n"
                    f"  URL: {direct_url}"
                )
                return False

            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            size = os.path.getsize(dest_path) if dest_path.exists() else 0
            if size > 1000:
                log.info(f"  Saved: {dest_path.name} ({round(size / 1_048_576, 2)} MB)")
                return True

            log.warning(f"  File too small ({size} bytes), retrying...")
            dest_path.unlink(missing_ok=True)

        except Exception as e:
            log.warning(f"  Attempt {attempt} error: {e}")
            time.sleep(2 * attempt)

    log.error(f"  FAILED after {MAX_RETRIES} attempts: {direct_url}")
    return False


def safe_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len]


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — PLAYWRIGHT: JS helpers for NPTEL transcript page
# ══════════════════════════════════════════════════════════════════════════════

ROW_FINDER_JS = """
function findRow(index) {
    const spans = document.querySelectorAll('span.c-name');
    if (index >= spans.length) return null;
    let cur = spans[index].parentElement;
    while (cur && cur !== document.body) {
        const tag = cur.tagName.toLowerCase();
        const cls = (cur.className || '').toLowerCase();
        if (tag === 'tr' ||
            cls.includes('row') || cls.includes('lecture') ||
            cls.includes('item') || cls.includes('card') ||
            cls.includes('content') || cls.includes('unit')) {
            return cur;
        }
        cur = cur.parentElement;
    }
    return spans[index].closest('li, div, tr') || spans[index].parentElement;
}
"""


def select_language(page, index: int, language: str) -> dict:
    """Open the language dropdown for row[index] and click the matching option."""
    return page.evaluate(f"""
    (function(index, language) {{
        {ROW_FINDER_JS}
        var row = findRow(index);
        if (!row) return {{lang_ok: false, msg: 'row not found'}};

        // Find and click the dropdown toggle
        var toggle = null;
        for (var el of row.querySelectorAll('*')) {{
            var txt = (el.innerText || el.textContent || '').trim().toLowerCase();
            if (txt.includes('select language') &&
                ['BUTTON','DIV','SPAN','A'].includes(el.tagName)) {{
                toggle = el; break;
            }}
        }}
        if (!toggle) return {{lang_ok: false, msg: 'toggle not found'}};

        toggle.click();
        toggle.dispatchEvent(new MouseEvent('mousedown', {{bubbles: true}}));

        // Find <li> options (may be in body via portal)
        var liOptions = [];
        for (var root of [row, document.body]) {{
            var lis = root.querySelectorAll('li');
            if (lis.length) {{ liOptions = Array.from(lis); break; }}
        }}

        var available = liOptions.map(li => (li.innerText || li.textContent || '').trim());
        var target = liOptions.find(li => {{
            var t = (li.innerText || li.textContent || '').trim().toLowerCase();
            return t === language.toLowerCase() || t.includes(language.toLowerCase());
        }});

        if (!target) return {{lang_ok: false, msg: 'option not found', available: available}};

        target.click();
        target.dispatchEvent(new MouseEvent('mousedown', {{bubbles: true}}));
        target.dispatchEvent(new MouseEvent('mouseup',   {{bubbles: true}}));
        return {{lang_ok: true, msg: 'selected',
                 chosen: (target.innerText || '').trim(), available: available}};
    }})({index}, '{language}')
    """)


def get_transcript_href(page, index: int):
    """Read the href/data-url of the Transcripts button for row[index]."""
    return page.evaluate(f"""
    (function(index) {{
        {ROW_FINDER_JS}
        var row = findRow(index);
        if (!row) return null;
        for (var el of row.querySelectorAll('a, button')) {{
            var txt = (el.innerText || el.textContent || '').trim().toLowerCase();
            if (txt === 'transcripts' || txt === 'transcript') {{
                return el.getAttribute('href') || el.getAttribute('data-url') ||
                       el.getAttribute('data-href') || '__needs_click__';
            }}
        }}
        return null;
    }})({index})
    """)


def click_transcript_and_intercept(page, index: int):
    """Click the Transcripts button and intercept the resulting PDF request."""
    try:
        with page.expect_request(
            lambda req: (
                ".pdf" in req.url.lower()
                or "transcript" in req.url.lower()
                or "download" in req.url.lower()
            ),
            timeout=12000,
        ) as req_info:
            page.evaluate(f"""
            (function(index) {{
                {ROW_FINDER_JS}
                var row = findRow(index);
                if (!row) return;
                for (var el of row.querySelectorAll('a, button')) {{
                    var txt = (el.innerText || el.textContent || '').trim().toLowerCase();
                    if (txt === 'transcripts' || txt === 'transcript') {{
                        el.click(); return;
                    }}
                }}
            }})({index})
            """)
        return req_info.value.url
    except Exception as e:
        log.warning(f"  Request intercept failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — MAIN TRANSCRIPT DOWNLOAD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def download_transcripts(
    session: requests.Session,
    course_url: str,
    download_dir: Path,
    language: str = LANGUAGE,
) -> list:
    results = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1440, "height": 1080},
            user_agent=HEADERS["User-Agent"],
        )

        try:
            log.info(f"Opening: {course_url}")
            page.goto(course_url, wait_until="domcontentloaded", timeout=60000)
            time.sleep(3)
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass

            # Click Downloads tab
            log.info("Clicking Downloads tab...")
            for fn in [
                lambda: page.get_by_role("tab",  name=re.compile(r"downloads", re.I)).first.click(timeout=5000),
                lambda: page.get_by_text(re.compile(r"^downloads$", re.I)).first.click(timeout=5000),
                lambda: page.get_by_role("link", name=re.compile(r"downloads", re.I)).first.click(timeout=5000),
            ]:
                try:
                    fn(); time.sleep(2); break
                except Exception:
                    continue

            # Expand Transcripts accordion
            log.info("Expanding Transcripts section...")
            try:
                page.get_by_text(re.compile(r"^transcripts$", re.I)).first.click(timeout=5000)
                time.sleep(2)
            except Exception:
                log.info("  Already open or not a button.")

            # Collect lecture rows
            all_spans = page.locator("span.c-name")
            lecture_indices = []
            for i in range(all_spans.count()):
                txt = all_spans.nth(i).inner_text().strip()
                if txt and txt.lower() != "chapter name":
                    lecture_indices.append((i, txt))

            log.info(f"Found {len(lecture_indices)} lectures.")

            # Process each lecture
            for rank, (i, lec_name) in enumerate(lecture_indices, 1):
                log.info(f"  [{rank}/{len(lecture_indices)}] {lec_name[:70]}")

                # Select language
                sel = select_language(page, i, language)
                if not sel or not sel.get("lang_ok"):
                    avail   = (sel or {}).get("available", [])
                    fallback = next((a for a in avail if "english" in a.lower()), None)
                    if fallback:
                        log.info(f"    Falling back to: '{fallback}'")
                        sel = select_language(page, i, fallback)
                    if not (sel and sel.get("lang_ok")):
                        log.warning(f"    Language selection failed: {(sel or {}).get('msg','?')}")
                        results.append({"lecture": lec_name, "url": None, "status": "lang_failed", "file": None})
                        continue

                log.info(f"    Language: {sel.get('chosen', language)}")
                time.sleep(0.5)

                # Get transcript URL
                href = get_transcript_href(page, i)
                if href and href != "__needs_click__":
                    pdf_url = href if href.startswith("http") else "https://nptel.ac.in" + href
                else:
                    pdf_url = click_transcript_and_intercept(page, i)

                if pdf_url:
                    fname = f"lec{rank}.pdf"
                    dest  = download_dir / fname
                    ok    = download_file(session, pdf_url, dest)
                    status = "downloaded" if ok else "failed"
                    log.info(f"    {'✔' if ok else '✘'} {fname}")
                    results.append({"lecture": lec_name, "url": pdf_url, "status": status, "file": str(dest)})
                else:
                    log.warning("    No URL obtained.")
                    results.append({"lecture": lec_name, "url": None, "status": "no_url", "file": None})

                time.sleep(0.4)

        finally:
            browser.close()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. Scrape and filter courses
    df     = scrape_courses()
    noc_df = filter_noc_cs(df)
    noc_df.to_csv(BASE_SAVE_DIR / "cs_nptel_courses_full.csv", index=False)

    session = requests.Session()
    session.headers.update(HEADERS)

    all_records = []

    for i in range(len(noc_df)):
        row        = noc_df.iloc[i]
        course_url = row["course_url"]
        course_id  = course_url.rstrip("/").split("/")[-1]
        save_dir   = BASE_SAVE_DIR / course_id
        save_dir.mkdir(parents=True, exist_ok=True)

        log.info("\n" + "=" * 60)
        log.info(f"Processing [{i+1}/{len(noc_df)}]: {course_id}")
        log.info("=" * 60)

        try:
            results = download_transcripts(
                session=session,
                course_url=course_url,
                download_dir=save_dir,
                language=LANGUAGE,
            )

            ok_count = sum(1 for r in results if r["status"] == "downloaded")
            log.info(f"  Downloaded {ok_count}/{len(results)} transcripts for {course_id}")

            for r in results:
                all_records.append({
                    "course_id":    course_id,
                    "course_title": row["course_title"],
                    "lecture":      r["lecture"],
                    "url":          r["url"],
                    "status":       r["status"],
                    "file":         r["file"],
                })

        except Exception as e:
            log.error(f"FAILED: {course_id} — {e}")
            all_records.append({
                "course_id":    course_id,
                "course_title": row["course_title"],
                "lecture":      None,
                "url":          None,
                "status":       "error",
                "file":         None,
            })

    # Save final summary
    summary_df = pd.DataFrame(all_records)
    out_path   = BASE_SAVE_DIR / "cs_nptel_with_lecture.csv"
    summary_df.to_csv(out_path, index=False)
    log.info(f"\nSummary saved to: {out_path}")
    log.info(f"Total records: {len(summary_df)}")


if __name__ == "__main__":
    main()