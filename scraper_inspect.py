"""
STEP 1 — Run this first.
Opens a real Chrome window so you can see the page and find selectors.
It prints the raw HTML of the first listing card so you can identify
the correct CSS classes to put into scraper.py.

Run with: .venv/Scripts/python scraper_inspect.py
"""
from playwright.sync_api import sync_playwright
import time

URL = "https://www.dubizzle.com/motors/cars/used-cars-for-sale/"

with sync_playwright() as p:
    # headless=False opens a visible Chrome window
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 900},
    )
    page = context.new_page()

    print(f"Opening {URL} ...")
    page.goto(URL, wait_until="networkidle", timeout=60000)

    # Wait for any listing cards to appear — adjust selector if needed
    # Common patterns: article, [data-testid*="listing"], li[class*="listing"]
    time.sleep(3)

    # Print the full URL in case of redirects
    print(f"Landed on: {page.url}")

    # Try to find listing cards using broad selectors
    for selector in [
        "article",
        "[data-testid*='listing']",
        "[class*='listing']",
        "li[class*='item']",
        "div[class*='card']",
    ]:
        elements = page.query_selector_all(selector)
        if elements:
            print(f"\nFound {len(elements)} elements with selector: {selector}")
            # Print the outer HTML of the first one so you can inspect it
            print("--- First element HTML (first 2000 chars) ---")
            print(elements[0].inner_html()[:2000])
            break
    else:
        print("\nNo listing cards found with standard selectors.")
        print("The page HTML (first 3000 chars):")
        print(page.content()[:3000])

    print("\nLeaving browser open for 30 seconds so you can inspect manually...")
    print("Right-click a listing card → Inspect → find the CSS class pattern")
    time.sleep(30)
    browser.close()
