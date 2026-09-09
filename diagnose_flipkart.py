import asyncio
from urllib.parse import quote_plus
from playwright.async_api import async_playwright

QUERY = "mobile phones"
URL = f"https://www.flipkart.com/search?q={quote_plus(QUERY)}"

CANDIDATE_SELECTORS = [
    {"label": "data-id cards", "card": "div[data-id]"},
    {"label": "_1AtVbE old layout", "card": "div._1AtVbE"},
    {"label": "slAVV8 layout", "card": "div.slAVV8"},
    {"label": "tUxRFH card", "card": "div.tUxRFH"},
    {"label": "DOjaWF card", "card": "div.DOjaWF"},
    {"label": "yhB1nd card", "card": "div.yhB1nd"},
    {"label": "_75nlfW card", "card": "div._75nlfW"},
    {"label": "a with /p/ href", "card": "a[href*='/p/']"},
]

NAME_SELECTORS = ["div.KzDlHZ","a.IRpwTa","div._4rR01T","div.wjcEIp","a.wjcEIp","a.s1Q9rs","a[title]","[class*='name']"]
PRICE_SELECTORS = ["div.Nx9bqj","div._30jeq3","div.D2rA8n","[class*='price']"]

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()
        print(f"[DIAG] Navigating to: {URL}")
        try:
            await page.goto(URL, wait_until="domcontentloaded", timeout=20000)
        except Exception as e:
            print(f"[DIAG] goto() raised: {e}")
        await page.wait_for_timeout(4000)
        title = await page.title()
        current_url = page.url
        print(f"[DIAG] Page title: {title!r}")
        print(f"[DIAG] Final URL:  {current_url!r}")
        print("\n--- Card selectors ---")
        for sel in CANDIDATE_SELECTORS:
            count = await page.locator(sel["card"]).count()
            print(f"  {sel['label']:35s}  count={count}")
        print("\n--- Name selectors ---")
        for ns in NAME_SELECTORS:
            count = await page.locator(ns).count()
            first = ""
            if count > 0:
                try: first = (await page.locator(ns).first.inner_text(timeout=1000))[:50]
                except: pass
            print(f"  {ns:35s}  count={count}  {first!r}")
        print("\n--- Price selectors ---")
        for ps in PRICE_SELECTORS:
            count = await page.locator(ps).count()
            first = ""
            if count > 0:
                try: first = (await page.locator(ps).first.inner_text(timeout=1000))[:40]
                except: pass
            print(f"  {ps:35s}  count={count}  {first!r}")
        print("\n--- Body text sample (first 2000 chars) ---")
        try:
            body_text = await page.locator("body").inner_text(timeout=5000)
            print(body_text[:2000])
        except Exception as e:
            print(f"ERROR: {e}")
        print("\n[DIAG] Waiting 15s - inspect browser window now...")
        await page.wait_for_timeout(15000)
        await context.close()
        await browser.close()
        print("[DIAG] Done.")

asyncio.run(main())
