"""
search.py
---------
The ONE function Person 3 needs: search_products(query, task_id).

Workflow:
  1. Validate the task is still current before doing any browser work.
  2. Open an isolated browser context/page for this task and register it
     with task_control so it can be force-closed on cancellation.
  3. Try Amazon.in first (primary target -- accessible, search results
     render without login, realistic Indian e-commerce demo).
     NOTE: Flipkart (original target) is unreachable from this machine --
     DNS/connection fails before any page loads (confirmed via diagnostic
     on 2026-09-08). Amazon.in loads reliably on the same network.
  4. If Amazon.in fails, fall back to a LOCAL demo page that serves
     query-matched products via a URL param (?q=...). This is still REAL
     browser automation (real page load + real DOM extraction) but every
     result carries a "source" tag so it is never mistaken for live data.
  5. Extract top products, applying client-side price/size filtering as a
     safety net on top of the site's own results.
  6. Re-check validity before returning/storing -- a task that was
     cancelled or superseded mid-flight must NEVER surface as current.

Every exit path returns a clean, JSON-serialisable dict shaped like:
    {"task_id": ..., "status": ..., "results": [...], ...}
Callers never need to catch a Playwright exception.

HOW TO RE-DIAGNOSE IF AMAZON BREAKS:
  Run backend/diagnose_flipkart.py (or a similar script) with HEADLESS=false,
  inspect browser window + printed selector counts. If count=0 everywhere,
  Amazon changed its DOM. Use DevTools > Inspect to find current class names
  for s-result-item, product title span, and price span.
"""

import os
import re
import logging
import random
from typing import Dict, Any
from urllib.parse import quote_plus

from playwright.async_api import TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

from .browser import browser_manager
from .task_control import registry
from .extraction import (
    extract_products_generic,
    filter_and_rank,
    parse_max_price,
    parse_size,
)

# How many ranked results to return. Raise via MAX_SEARCH_RESULTS env var.
# Must be <= extract_products_generic's max_cards (currently 15); if you
# raise this above 15 also raise max_cards in extraction.py to match.
MAX_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))


logger = logging.getLogger(__name__)

# Primary: Amazon.in -- accessible from this network when Flipkart is not
AMAZON_SEARCH_URL = "https://www.amazon.in/s?k={query}&ref=nb_sb_noss"

# Legacy / secondary attempt (often unreachable -- kept for completeness)
FLIPKART_SEARCH_URL = "https://www.flipkart.com/search?q={query}"

FALLBACK_PAGE_PATH = os.path.join(os.path.dirname(__file__), "fallback_page.html")


def _clean_error(task_id: str, message: str) -> Dict[str, Any]:
    return {"task_id": task_id, "status": "error", "error": message, "results": []}


def _stale(task_id: str) -> Dict[str, Any]:
    return {"task_id": task_id, "status": "stale", "results": []}


def _build_clean_query(query: str) -> str:
    """Strip price constraints and conversational filler; keep product keywords."""
    clean = re.sub(r"under\s*(?:Rs\.?|INR|₹)?\s*\d+", "", query, flags=re.IGNORECASE)
    clean = re.sub(r"below\s*(?:Rs\.?|INR|₹)?\s*\d+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"size\s*\d+", "", clean, flags=re.IGNORECASE)
    words = re.findall(r"[a-zA-Z0-9]+", clean.lower())
    stopwords = {
        "a", "an", "the", "and", "or", "in", "of", "for", "to", "is", "it",
        "that", "this", "find", "search", "show", "me", "some", "get", "want",
        "finally", "please", "help", "good", "best", "nice", "buy", "looking", "need",
        "can", "you", "i", "my", "with", "what", "are", "there", "any", "give",
        "suggest", "tell", "hi", "hello", "okay", "ok",
    }
    return " ".join([w for w in words if w not in stopwords]) or query


async def _run_amazon_search(page, query: str):
    """
    Navigate to Amazon.in and extract products.

    Amazon.in CSS selectors (verified 2025-09):
      Card:  div[data-component-type="s-search-result"]
             (attribute-based -- very stable, Amazon has used this for years)
      Name:  h2.a-size-mini span.a-text-normal  OR  h2 span
             (structural -- also stable)
      Price: span.a-price-whole  OR  span.a-offscreen
      Link:  h2 a  (relative href starting with /)
      Image: img.s-image

    Fallback selector set uses broader attribute/structural patterns that
    survive minor Amazon A/B layout changes.
    """
    clean_q = _build_clean_query(query)
    url = AMAZON_SEARCH_URL.format(query=quote_plus(clean_q))

    logger.info("[amazon] Searching: %r -> URL: %s", clean_q, url)

    # Brief delay before navigation (looks less robotic)
    await page.wait_for_timeout(random.randint(200, 400))

    await page.goto(url, wait_until="domcontentloaded", timeout=20000)

    # Brief post-load wait for price widgets to settle
    await page.wait_for_timeout(300)


    title = await page.title()
    logger.info("[amazon] Page title: %r | URL: %r", title, page.url)

    # Bail out early if Amazon showed a captcha or login page
    if any(kw in title.lower() for kw in ("robot", "captcha", "sign in", "verify")):
        logger.warning("[amazon] Bot/captcha/login wall detected, aborting Amazon search")
        return []

    # Primary selector set -- attribute-based card identifier (very stable)
    # LINK NOTE: Amazon's h2 a element exists but has an empty href attribute --
    # the actual product URL lives on a.a-link-normal[href*='/dp/'] inside the card.
    # This was confirmed via [href-debug] logging showing hrefRaw='' for h2 a.
    selector_sets = [
        {
            "card":  'div[data-component-type="s-search-result"]',
            "name":  "h2.a-size-mini span.a-text-normal, h2 span.a-text-normal, h2 span",
            "price": "span.a-price-whole, span.a-offscreen",
            "link":  "a.a-link-normal[href*='/dp/'], a[href*='/dp/'], h2 a",
            "image": "img.s-image",
        },
        # Broader fallback for A/B layout variants
        {
            "card":  "div.s-result-item[data-asin]",
            "name":  "span.a-text-normal, h2 span",
            "price": "span.a-price-whole, span.a-offscreen",
            "link":  "a.a-link-normal[href*='/dp/'], a[href*='/dp/'], h2 a",
            "image": "img.s-image, img[srcset]",
        },
    ]


    for sel in selector_sets:
        try:
            card_count = await page.locator(sel["card"]).count()
            logger.info("[amazon] Selector %r matched %d cards", sel["card"][:50], card_count)
            if card_count == 0:
                continue
        except PlaywrightTimeoutError:
            continue

        products = await extract_products_generic(
            page,
            card_selector=sel["card"],
            name_selector=sel["name"],
            price_selector=sel["price"],
            link_selector=sel["link"],
            image_selector=sel["image"],
        )
        logger.info("[amazon] Extracted %d products from selector set", len(products))
        if products:
            # Convert relative Amazon hrefs to absolute
            for p in products:
                if p.get("url") and p["url"].startswith("/"):
                    p["url"] = "https://www.amazon.in" + p["url"]
            return products

    logger.warning("[amazon] No products extracted from any selector set")
    return []


async def _run_flipkart_search(page, query: str):
    """
    SECONDARY / LEGACY: Flipkart search.
    NOTE: Flipkart is unreachable from this machine (DNS/connection failure,
    confirmed 2026-09-08). This function is kept as a secondary attempt in
    case network conditions change. If Flipkart loads, the selectors below
    were last valid circa 2024 and may need refreshing via DevTools inspect.
    """
    clean_q = _build_clean_query(query)
    url = FLIPKART_SEARCH_URL.format(query=quote_plus(clean_q))
    logger.info("[flipkart] Attempting: %s", url)

    await page.goto(url, wait_until="domcontentloaded", timeout=10000)
    await page.wait_for_timeout(1500)

    title = await page.title()
    logger.info("[flipkart] Page title: %r", title)

    if "chrome-error" in page.url or not title or "Loading" in title:
        logger.warning("[flipkart] Page failed to load (chrome-error or stuck loading)")
        return []

    # Try to close login popup
    try:
        close_btn = page.locator("button._2KpZ6l._2doB4z")
        if await close_btn.count() > 0:
            await close_btn.first.click(timeout=2000)
    except Exception:
        pass

    selector_sets = [
        {
            "card":  "div[data-id]",
            "name":  "div.KzDlHZ, a.wjcEIp, a.IRpwTa, div._4rR01T, a.s1Q9rs, a[title]",
            "price": "div.Nx9bqj, div._30jeq3, div.D2rA8n",
            "link":  "a",
            "image": "img",
        },
        {
            "card":  "div._1AtVbE",
            "name":  "div._4rR01T, a.s1Q9rs, a.IRpwTa",
            "price": "div._30jeq3",
            "link":  "a",
            "image": "img",
        },
    ]

    for sel in selector_sets:
        try:
            await page.wait_for_selector(sel["card"], timeout=3000)
        except PlaywrightTimeoutError:
            continue

        products = await extract_products_generic(
            page,
            card_selector=sel["card"],
            name_selector=sel["name"],
            price_selector=sel["price"],
            link_selector=sel["link"],
            image_selector=sel["image"],
        )
        if products:
            return products

    return []


async def _run_fallback_search(page, query: str):
    """
    Local, clearly-labelled fallback. Real browser automation against a
    real local page (not a hardcoded Python dict) -- but explicitly NOT
    live web data, and every result carries a "source" tag saying so.

    The local page accepts ?q=QUERY to filter cards to the right category.
    """
    clean_q = _build_clean_query(query)
    file_url = "file://" + FALLBACK_PAGE_PATH + "?q=" + quote_plus(clean_q)
    await page.goto(file_url, wait_until="domcontentloaded")
    await page.wait_for_timeout(500)  # let JS filter run

    products = await extract_products_generic(
        page,
        card_selector=".product-card",
        name_selector=".product-name",
        price_selector=".product-price",
        link_selector="a.product-link",
        image_selector="img.product-image",
    )
    for p in products:
        p["source"] = "local_fallback_demo_page"
        if not p.get("url") or "example.com" in p.get("url", ""):
            p["url"] = f"https://www.amazon.in/s?k={quote_plus(p['name'])}"
    return products


async def search_products(query: str, task_id: str) -> Dict[str, Any]:
    """
    Real-browser product search with task versioning built in.

    IMPORTANT: the caller must register the task BEFORE calling this, e.g.:
        registry.create_task(task_id, query)
        result = await search_products(query, task_id)
    create_task() is what invalidates any previous "current" task -- this
    function only checks/respects validity, it does not create tasks.

    Args:
        query: natural language query, e.g. "black running shoes under 1500 size 9"
        task_id: unique id for this task.

    Returns a JSON-serialisable dict, always shaped like:
        {"task_id": ..., "status": ..., "results": [...], ...}
    """
    if not registry.is_task_valid(task_id):
        return _stale(task_id)

    registry.mark_running(task_id)

    max_price = parse_max_price(query)
    size = parse_size(query)

    context = None
    page = None
    source_used = "amazon"

    try:
        context = await browser_manager.new_context()
        page = await context.new_page()
        registry.attach_page(task_id, page)

        if not registry.is_task_valid(task_id):
            return _stale(task_id)

        raw_products = []
        primary_err = None

        # --- ATTEMPT 1: Amazon.in (primary) ---
        try:
            raw_products = await _run_amazon_search(page, query)
            if not raw_products:
                raise RuntimeError("no_results_from_amazon")
            source_used = "amazon"
        except (PlaywrightTimeoutError, PlaywrightError, RuntimeError) as err1:
            logger.warning("[search] Amazon failed: %s -- trying Flipkart", err1)
            primary_err = err1

            if not registry.is_task_valid(task_id):
                return _stale(task_id)

            # --- ATTEMPT 2: Flipkart (secondary, usually unreachable) ---
            try:
                raw_products = await _run_flipkart_search(page, query)
                if not raw_products:
                    raise RuntimeError("no_results_from_flipkart")
                source_used = "flipkart"
            except (PlaywrightTimeoutError, PlaywrightError, RuntimeError) as err2:
                logger.warning("[search] Flipkart also failed: %s -- using local fallback", err2)

                if not registry.is_task_valid(task_id):
                    return _stale(task_id)

                # --- ATTEMPT 3: Local fallback (always works) ---
                source_used = "local_fallback_demo_page"
                try:
                    raw_products = await _run_fallback_search(page, query)
                except Exception as fallback_err:
                    registry.mark_error(task_id, str(fallback_err))
                    return _clean_error(
                        task_id,
                        f"all_sources_failed: amazon={primary_err}; flipkart={err2}; fallback={fallback_err}",
                    )

        if not registry.is_task_valid(task_id):
            return _stale(task_id)

        ranked = filter_and_rank(raw_products, max_price=max_price, size=size, top_n=MAX_RESULTS, query=query)

        result: Dict[str, Any] = {
            "task_id": task_id,
            "status": "completed",
            "source": source_used,
            "query": query,
            "parsed_constraints": {"max_price": max_price, "size": size},
            "results": ranked,
        }
        if source_used == "local_fallback_demo_page":
            result["warning"] = (
                "Live e-commerce sites were unreachable or blocked automation. "
                "These results are from a LOCAL FALLBACK DEMO PAGE, not live web data."
            )
        if not ranked:
            result["status"] = "completed_empty"
            result["note"] = "No products matched the given constraints."

        if not registry.is_task_valid(task_id):
            return _stale(task_id)

        registry.mark_completed(task_id, result)
        return result

    except PlaywrightError as e:
        if not registry.is_task_valid(task_id):
            return _stale(task_id)
        registry.mark_error(task_id, str(e))
        return _clean_error(task_id, f"browser_error: {e}")

    except Exception as e:
        if not registry.is_task_valid(task_id):
            return _stale(task_id)
        registry.mark_error(task_id, str(e))
        return _clean_error(task_id, f"unexpected_error: {e}")

    finally:
        try:
            if page and not page.is_closed():
                await page.close()
        except Exception:
            pass
        try:
            if context:
                await context.close()
        except Exception:
            pass
