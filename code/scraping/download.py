# ──────────────────────────────────────────────────────────────
# NPTEL Transcript Downloader — FIXED v3
# Key fixes:
#   1. Language is "English-Verified" not "English"
#   2. Custom Svelte dropdown: select option via JS click (no focus loss)
#   3. Transcript button also clicked via JS in same evaluate call
# ──────────────────────────────────────────────────────────────

"""
!pip install -q playwright requests
!playwright install chromium
!apt-get install -y chromium-driver > /dev/null 2>&1
"""

import re
import time
import requests
from pathlib import Path
from playwright.sync_api import sync_playwright

COURSE_URL   = "https://nptel.ac.in/courses/101101079"
LANGUAGE     = "English-Verified"       # ← exact text shown in dropdown
#DOWNLOAD_DIR = Path("transcripts_english")
COURSE_ID = COURSE_URL.rstrip("/").split("/")[-1]
DOWNLOAD_DIR = Path(COURSE_ID)
HEADERS      = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

DOWNLOAD_DIR.mkdir(exist_ok=True)
session = requests.Session()
session.headers.update(HEADERS)


def safe_filename(text, max_len=80):
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    text = re.sub(r'\s+', "_", text.strip())
    return text[:max_len]


def download_file(url, dest_path):
    try:
        r = session.get(url, timeout=30, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    ❌ Download failed: {e}")
        return False


# ── Core JS helper: finds row for nth span.c-name ────────────
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
    // fallback: return parent of the span itself
    return spans[index].closest('li, div, tr') || spans[index].parentElement;
}
"""


def select_and_get_url(page, index, language):
    """
    Single JS evaluate call that:
      1. Opens the custom Svelte dropdown for row[index]
      2. Immediately clicks the matching <li> option (no Python round-trip = no focus loss)
      3. Reads the href/data-url of the Transcripts button
    Returns dict: {lang_ok, href, available_langs}
    """
    result = page.evaluate(f"""
    (function(index, language) {{
        {ROW_FINDER_JS}

        var row = findRow(index);
        if (!row) return {{lang_ok: false, msg: 'row not found'}};

        // ── Step 1: Open dropdown ─────────────────────────────
        // The toggle is the div/button with "Select Language" text
        var toggle = null;
        var allEls = row.querySelectorAll('*');
        for (var el of allEls) {{
            var txt = (el.innerText || el.textContent || '').trim().toLowerCase();
            if ((txt === 'select language' || txt.includes('select language')) &&
                (el.tagName === 'BUTTON' || el.tagName === 'DIV' ||
                 el.tagName === 'SPAN'   || el.tagName === 'A')) {{
                toggle = el;
                break;
            }}
        }}

        if (!toggle) return {{lang_ok: false, msg: 'toggle not found'}};

        // Click to open
        toggle.click();
        toggle.dispatchEvent(new MouseEvent('mousedown', {{bubbles: true}}));

        // ── Step 2: Find <li> options NOW (while open) ───────
        // They may be inside the row OR appended to body (portals)
        var searchRoots = [row, document.body];
        var liOptions = [];
        for (var root of searchRoots) {{
            var lis = root.querySelectorAll('li');
            if (lis.length > 0) {{
                liOptions = Array.from(lis);
                break;
            }}
        }}

        var available = liOptions.map(li => (li.innerText || li.textContent || '').trim());
        var target = liOptions.find(li => {{
            var t = (li.innerText || li.textContent || '').trim().toLowerCase();
            return t === language.toLowerCase() || t.includes(language.toLowerCase());
        }});

        if (!target) {{
            return {{lang_ok: false, msg: 'option not found', available: available}};
        }}

        // Click it immediately (same JS thread — dropdown stays open)
        target.click();
        target.dispatchEvent(new MouseEvent('mousedown', {{bubbles: true}}));
        target.dispatchEvent(new MouseEvent('mouseup',   {{bubbles: true}}));

        return {{lang_ok: true, msg: 'selected', chosen: (target.innerText||'').trim(),
                 available: available}};
    }})({index}, '{language}')
    """)

    return result


def get_transcript_href(page, index):
    """
    Reads the href/data-url of the Transcripts button for row[index] via JS.
    Returns URL string or None.
    """
    return page.evaluate(f"""
    (function(index) {{
        {ROW_FINDER_JS}
        var row = findRow(index);
        if (!row) return null;
        var els = row.querySelectorAll('a, button');
        for (var el of els) {{
            var txt = (el.innerText || el.textContent || '').trim().toLowerCase();
            if (txt === 'transcripts' || txt === 'transcript') {{
                return el.getAttribute('href') || el.getAttribute('data-url') ||
                       el.getAttribute('data-href') || '__needs_click__';
            }}
        }}
        return null;
    }})({index})
    """)


def click_transcript_and_intercept(page, index):
    """
    Clicks the Transcripts button via JS and intercepts the resulting PDF request.
    Returns URL string or None.
    """
    try:
        with page.expect_request(
            lambda req: (
                ".pdf" in req.url.lower() or
                "transcript" in req.url.lower() or
                "download" in req.url.lower()
            ),
            timeout=12000
        ) as req_info:
            page.evaluate(f"""
            (function(index) {{
                {ROW_FINDER_JS}
                var row = findRow(index);
                if (!row) return;
                var els = row.querySelectorAll('a, button');
                for (var el of els) {{
                    var txt = (el.innerText || el.textContent || '').trim().toLowerCase();
                    if (txt === 'transcripts' || txt === 'transcript') {{
                        el.click();
                        return;
                    }}
                }}
            }})({index})
            """)
        return req_info.value.url
    except Exception as e:
        print(f"    ⚠️  Request intercept failed: {e}")
        return None


def download_transcripts(course_url, language=LANGUAGE):
    results = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1440, "height": 1080},
            user_agent=HEADERS["User-Agent"]
        )

        try:
            # ── Step 1: Load page ─────────────────────────────
            print(f"🌐 Opening {course_url}")
            page.goto(course_url, wait_until="domcontentloaded", timeout=60000)
            time.sleep(3)
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass

            # ── Step 2: Click Downloads tab ───────────────────
            print("📂 Clicking Downloads tab...")
            for fn in [
                lambda: page.get_by_role("tab",   name=re.compile(r"downloads", re.I)).first.click(timeout=5000),
                lambda: page.get_by_text(re.compile(r"^downloads$", re.I)).first.click(timeout=5000),
                lambda: page.get_by_role("link",  name=re.compile(r"downloads", re.I)).first.click(timeout=5000),
            ]:
                try:
                    fn(); time.sleep(2); break
                except Exception:
                    continue

            # ── Step 3: Expand Transcripts accordion ──────────
            print("📖 Expanding Transcripts...")
            try:
                page.get_by_text(re.compile(r"^transcripts$", re.I)).first.click(timeout=5000)
                time.sleep(2)
            except Exception:
                print("  ℹ️  Already open or not a button")

            # ── Step 4: Get all lecture rows ───────────────────
            all_spans = page.locator("span.c-name")
            lecture_indices = []
            for i in range(all_spans.count()):
                txt = all_spans.nth(i).inner_text().strip()
                if txt and txt.lower() != "chapter name":
                    lecture_indices.append((i, txt))

            print(f"📋 Found {len(lecture_indices)} lectures\n")

            # ── Step 5: Process each lecture ──────────────────
            for rank, (i, lec_name) in enumerate(lecture_indices, 1):
                print(f"  [{rank}/{len(lecture_indices)}] {lec_name[:70]}")

                # 5a. Select language via JS (dropdown stays open)
                sel_result = select_and_get_url(page, i, language)

                if not sel_result or not sel_result.get("lang_ok"):
                    avail = sel_result.get("available", []) if sel_result else []
                    print(f"    ⚠️  Language not set — {sel_result.get('msg','?')}")
                    if avail:
                        print(f"         Available: {avail}")
                        # Auto-fallback: try first English-like option
                        fallback = next((a for a in avail if "english" in a.lower()), None)
                        if fallback:
                            print(f"         Retrying with fallback: '{fallback}'")
                            sel_result = select_and_get_url(page, i, fallback)
                            if not (sel_result and sel_result.get("lang_ok")):
                                results.append({"lecture": lec_name, "url": None,
                                                "status": "lang_failed", "file": None})
                                continue
                        else:
                            results.append({"lecture": lec_name, "url": None,
                                            "status": "lang_failed", "file": None})
                            continue

                chosen = sel_result.get("chosen", language)
                print(f"    ✅ Language: {chosen}")
                time.sleep(0.5)   # let Svelte re-render the Transcripts button href

                # 5b. Get transcript URL
                href = get_transcript_href(page, i)

                if href and href != "__needs_click__":
                    pdf_url = href if href.startswith("http") else "https://nptel.ac.in" + href
                else:
                    # Href not in DOM — intercept via network request
                    pdf_url = click_transcript_and_intercept(page, i)

                if pdf_url:
                    #fname   = f"{rank:03d}_{safe_filename(lec_name)}.pdf"
                    fname = f"lec{rank}.pdf"
                    dest    = DOWNLOAD_DIR / fname
                    ok      = download_file(pdf_url, dest)
                    status  = "downloaded" if ok else "failed"
                    print(f"    {'💾' if ok else '❌'} {fname}")
                    results.append({"lecture": lec_name, "url": pdf_url,
                                    "status": status, "file": str(dest)})
                else:
                    print(f"    ❌ No URL obtained")
                    results.append({"lecture": lec_name, "url": None,
                                    "status": "no_url", "file": None})

                time.sleep(0.4)

        finally:
            browser.close()

    return results


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"  NPTEL Transcript Downloader  |  {LANGUAGE}")
    print(f"  {COURSE_URL}")
    print("=" * 60)

    results = download_transcripts(COURSE_URL)

    ok   = [r for r in results if r["status"] == "downloaded"]
    fail = [r for r in results if r["status"] != "downloaded"]

    print("\n" + "=" * 60)
    print(f"  ✅ Downloaded : {len(ok)}")
    print(f"  ❌ Failed     : {len(fail)}")
    print(f"  📁 Saved to   : {DOWNLOAD_DIR.resolve()}/")
    print("=" * 60)

    if fail:
        print("\nFailed lectures:")
        for r in fail:
            print(f"  • {r['lecture']} — {r['status']}")