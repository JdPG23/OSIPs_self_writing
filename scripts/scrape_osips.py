"""
Scraper for ESA OSIP implemented ideas.
Fetches publicly available OSIP data to populate the corpus.

Usage:
    python scripts/scrape_osips.py
    python scripts/scrape_osips.py --year 2024
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import urllib.request
    from html.parser import HTMLParser
except ImportError:
    pass

# Output directory
REFS_DIR = Path(__file__).parent.parent / "corpus" / "references"


class SimpleHTMLTextExtractor(HTMLParser):
    """Minimal HTML to text converter."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = False
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"):
            self.text_parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.text_parts.append(data)

    def get_text(self):
        return "".join(self.text_parts)


def fetch_url(url: str) -> str:
    """Fetch URL content as text."""
    req = urllib.request.Request(url, headers={"User-Agent": "OSIP-Research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_text(html: str) -> str:
    """Extract readable text from HTML."""
    parser = SimpleHTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def parse_osip_list(text: str) -> list[dict]:
    """Extract OSIP entries from page text. Best-effort parsing."""
    entries = []
    # Look for patterns like title lines followed by institution/country info
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    current = {}
    for line in lines:
        # Heuristic: lines that look like OSIP titles (all caps acronym or
        # specific patterns)
        if re.match(r'^[A-Z][A-Z0-9\-]{2,}[\s:–—-]', line) or \
           (len(line) > 20 and len(line) < 200 and not line.startswith("©")):
            if current.get("title"):
                entries.append(current)
            current = {"title": line}
        elif current.get("title") and not current.get("institution"):
            current["institution"] = line

    if current.get("title"):
        entries.append(current)

    return entries


def save_entries(entries: list[dict], year: str):
    """Save parsed entries as JSON files."""
    REFS_DIR.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(entries):
        filename = f"osip_{year}_{i+1:03d}.json"
        filepath = REFS_DIR / filename
        entry["year"] = int(year)
        entry["status"] = "implemented"
        entry["source"] = "esa.int"
        filepath.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {len(entries)} entries to {REFS_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Scrape ESA OSIP implemented ideas")
    parser.add_argument("--year", type=str, default="2024", help="Year to scrape")
    parser.add_argument("--dry-run", action="store_true", help="Print without saving")
    args = parser.parse_args()

    # Known ESA OSIP pages (update URLs as ESA publishes new ones)
    urls = {
        "2023": "https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Shaping_the_Future/Implemented_OSIP_ideas_2023",
        "2024": "https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Shaping_the_Future/Implemented_OSIP_ideas_2024",
        "2025": "https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Shaping_the_Future/Implemented_OSIP_ideas_2025",
    }

    url = urls.get(args.year)
    if not url:
        print(f"No known URL for year {args.year}. Known years: {list(urls.keys())}")
        sys.exit(1)

    print(f"Fetching {url}...")
    try:
        html = fetch_url(url)
        text = extract_text(html)
        entries = parse_osip_list(text)
        print(f"Found {len(entries)} potential OSIP entries")

        if args.dry_run:
            for e in entries[:10]:
                print(f"  - {e.get('title', '?')[:80]}")
        else:
            save_entries(entries, args.year)

    except Exception as e:
        print(f"Error: {e}")
        print("You may need to manually populate corpus/references/ with OSIP data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
