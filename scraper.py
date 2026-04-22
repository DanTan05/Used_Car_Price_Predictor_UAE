"""
Cars24 UAE scraper — extracts used car data from cars24.ae
Output: cars24_raw.csv

Run: .venv/Scripts/python scraper.py

How it works:
  Phase 1 — paginate through search results, collect listing URLs
  Phase 2 — visit each listing, extract JSON from window.__PRELOADED_STATE__
  Saves after every row so progress is never lost if interrupted
"""
import csv
import json
import re
import time
import random
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
SEARCH_URL  = "https://www.cars24.ae/buy-used-cars-dubai/?page={page}"
OUTPUT_FILE = Path(__file__).parent / "cars24_raw.csv"
MAX_PAGES   = 65        # ~25 listings/page × 65 pages ≈ 1,600 cars
DELAY_MIN   = 1.5
DELAY_MAX   = 3.0
HEADERS     = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

FIELDS = [
    "url", "make", "model", "year", "trim", "price_aed", "original_price_aed",
    "mileage_km", "body_type", "transmission", "fuel_type", "engine_size_l",
    "cylinders", "drive_type", "color", "interior_trim", "specs_region",
    "warranty_months", "accident_free", "owner_number",
    "city", "listing_id",
]
# ─────────────────────────────────────────────────────────────────────────────


def get(url):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.text
        except Exception as e:
            print(f"    Attempt {attempt+1} failed ({e}), retrying...")
            time.sleep(3)
    return None


def random_delay():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


def collect_listing_urls(html):
    # Match only proper listing URLs: /buy-used-<make>-<model>-<year>-cars-dubai-<id>/
    # Year is 4 digits, ID is 10 digits — this avoids matching category/brand pages
    pattern = r'href="((?:https://www\.cars24\.ae)?/buy-used-[a-z0-9-]+-\d{4}-cars-dubai-\d{8,}/)"'
    urls = []
    for href in re.findall(pattern, html):
        if not href.startswith("http"):
            href = "https://www.cars24.ae" + href
        urls.append(href)
    return list(dict.fromkeys(urls))


def extract_state(html):
    """Pull window.__PRELOADED_STATE__ JSON out of the page."""
    m = re.search(r'__PRELOADED_STATE__\s*=\s*(\{.+?\});?\s*</script>', html, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def parse_listing(url, html):
    state = extract_state(html)
    if not state:
        return None

    c = state.get("carDetails", {}).get("content", {})
    if not c:
        return None

    return {
        "url":                  url,
        "make":                 c.get("make", ""),
        "model":                c.get("model", ""),
        "year":                 c.get("year", ""),
        "trim":                 c.get("variant", ""),
        "price_aed":            c.get("price", ""),
        "original_price_aed":   c.get("targetPrice", ""),
        "mileage_km":           c.get("odometerReading", ""),
        "body_type":            c.get("bodyType", ""),
        "transmission":         c.get("transmissionType", ""),
        "fuel_type":            c.get("fuelType", ""),
        "engine_size_l":        c.get("engineSize", ""),
        "cylinders":            c.get("noOfCylinders", ""),
        "drive_type":           c.get("driveType", ""),
        "color":                c.get("color", ""),
        "interior_trim":        c.get("interiorTrimType", ""),
        "specs_region":         c.get("specs", ""),
        "warranty_months":      c.get("warrantyDuration", ""),
        "accident_free":        c.get("accidentFree", ""),
        "owner_number":         c.get("ownerNumber", ""),
        "city":                 c.get("city", ""),
        "listing_id":           c.get("appointmentId", ""),
    }


def main():
    # Phase 1 — collect listing URLs
    print("Phase 1: collecting listing URLs...")
    all_urls = []

    for page_num in range(1, MAX_PAGES + 1):
        html = get(SEARCH_URL.format(page=page_num))
        if not html:
            print(f"  Page {page_num}: failed, stopping.")
            break

        urls = collect_listing_urls(html)
        if not urls:
            print(f"  Page {page_num}: no listings found, stopping.")
            break

        all_urls.extend(urls)
        print(f"  Page {page_num}: {len(urls)} listings (running total: {len(all_urls)})")
        random_delay()

    all_urls = list(dict.fromkeys(all_urls))
    print(f"\nTotal unique listings to scrape: {len(all_urls)}")

    # Resume support — skip already-scraped URLs
    scraped = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scraped.add(row.get("url", ""))
        print(f"Resuming — {len(scraped)} already done, skipping.")

    remaining = [u for u in all_urls if u not in scraped]
    print(f"Phase 2: scraping {len(remaining)} listings...\n")

    mode = "a" if scraped else "w"
    with open(OUTPUT_FILE, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not scraped:
            writer.writeheader()

        for i, url in enumerate(remaining, 1):
            print(f"  [{i:4d}/{len(remaining)}] {url[25:75]}")
            html = get(url)
            if html:
                row = parse_listing(url, html)
                if row:
                    writer.writerow(row)
                    f.flush()
                else:
                    print("         Could not parse listing")
            random_delay()

    print(f"\nDone. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
