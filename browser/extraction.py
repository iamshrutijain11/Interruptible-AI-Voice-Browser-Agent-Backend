"""
extraction.py
-------------
Turns raw Playwright DOM content into clean structured product dicts, and
provides simple constraint parsing/filtering used to apply "under â‚¹X" /
"size N" on top of whatever the target site's own search returns.

Kept free of task_control/browser imports so it's easy to unit test in
isolation (extract_products_generic needs a live Page, but the parsing/
filtering functions are pure and testable with no browser at all).
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)



def parse_max_price(query: str) -> Optional[int]:
    text = re.sub(r"(\d+),(\d+)", r"\1\2", query.lower())
    patterns = [
        r"under\s*(?:â‚¹|rs\.?|inr)?\s*(\d{2,8})",
        r"below\s*(?:â‚¹|rs\.?|inr)?\s*(\d{2,8})",
        r"(?:â‚¹|rs\.?|inr)\s*(\d{2,8})\s*(?:or less|max)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return None


def parse_size(query: str) -> Optional[str]:
    """Extract a shoe size like 'size 9' -> '9'."""
    m = re.search(r"size\s*(\d{1,2}(?:\.\d)?)", query.lower())
    return m.group(1) if m else None


def price_to_int(price_text: str) -> Optional[int]:
    """'â‚¹1,299' -> 1299. Returns None if unparseable."""
    if not price_text:
        return None
    digits = re.sub(r"[^\d]", "", price_text)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


async def _safe_text(locator) -> str:
    try:
        txt = await locator.inner_text(timeout=2000)
        return txt.strip()
    except Exception:
        return ""


async def _safe_attr(locator, attr: str) -> str:
    try:
        val = await locator.get_attribute(attr, timeout=2000)
        return val or ""
    except Exception:
        return ""


async def extract_products_generic(
    page,
    card_selector: str,
    name_selector: str,
    price_selector: str,
    link_selector: str,
    image_selector: str,
    max_cards: int = 15,   # must be >= MAX_SEARCH_RESULTS in search.py (default 10)
) -> List[Dict[str, Any]]:
    """
    Generic card-based extractor using a single page.evaluate() call for speed.

    Instead of Python-side sequential awaits (4 round-trips per card = slow),
    one JavaScript snippet reads ALL cards at once and returns the full array.
    This reduces ~30s sequential extraction to under 2s.

    Failures on individual cards are handled inside the JS (return null and
    filtered out), so one broken/ad card never crashes the whole extraction.
    """
    try:
        raw = await page.evaluate(
            """([cardSel, nameSel, priceSel, linkSel, imageSel, maxCards, pageUrl]) => {
                const cards = Array.from(document.querySelectorAll(cardSel)).slice(0, maxCards);
                let origin = '';
                try { origin = new URL(pageUrl).origin; } catch(e) {}
                return cards.map(card => {
                    try {
                        let name = '';
                        for (const ns of nameSel.split(',').map(s => s.trim())) {
                            const el = card.querySelector(ns);
                            if (el && el.textContent.trim()) { name = el.textContent.trim(); break; }
                        }
                        let price = '';
                        for (const ps of priceSel.split(',').map(s => s.trim())) {
                            const el = card.querySelector(ps);
                            if (el && el.textContent.trim()) { price = el.textContent.trim(); break; }
                        }
                        if (!name || !price) return null;
                        let href = '';
                        let hrefRaw = '';  // raw value before abs conversion, for debug logging
                        for (const ls of linkSel.split(',').map(s => s.trim())) {
                            const el = card.querySelector(ls);
                            if (el && el.getAttribute('href')) {
                                hrefRaw = el.getAttribute('href');
                                href = hrefRaw;
                                break;
                            }
                        }
                        // Robust fallbacks: some Amazon cards use /gp/, redirect links, or different anchor structures
                        if (!href) {
                            const h2a = card.querySelector('h2 a, a:has(h2)');
                            if (h2a && h2a.getAttribute('href')) href = h2a.getAttribute('href');
                        }
                        if (!href) {
                            const anyProd = card.querySelector("a[href*='/dp/'], a[href*='/gp/'], a.a-link-normal[href]");
                            if (anyProd && anyProd.getAttribute('href')) href = anyProd.getAttribute('href');
                        }
                        if (!href) {
                            const anyA = Array.from(card.querySelectorAll('a[href]')).find(a => {
                                const h = a.getAttribute('href') || '';
                                return h && h !== '#' && !h.startsWith('javascript:');
                            });
                            if (anyA) href = anyA.getAttribute('href');
                        }
                        if (href && href.startsWith('/') && origin) href = origin + href;

                        let image = '';
                        for (const is_ of imageSel.split(',').map(s => s.trim())) {
                            const el = card.querySelector(is_);
                            if (el) { image = el.getAttribute('src') || el.getAttribute('data-src') || ''; if (image) break; }
                        }
                        return { name, price, href, hrefRaw, image };
                    } catch(e) { return null; }
                }).filter(x => x !== null);
            }""",
            [card_selector, name_selector, price_selector, link_selector,
             image_selector, max_cards, page.url],
        )
    except Exception:
        return []

    results: List[Dict[str, Any]] = []
    for i, item in enumerate(raw or []):
        try:
            # Log raw href for first 3 cards so click-through issues are visible in backend output
            if i < 3:
                logger.info(
                    "[href-debug] card[%d] linkSel=%r hrefRaw=%r -> href=%r",
                    i, link_selector, item.get("hrefRaw", ""), item.get("href", ""),
                )
            results.append({
                "name":        item["name"],
                "price":       item["price"],
                "price_value": price_to_int(item["price"]),
                "url":         item["href"] or None,
                "image":       item["image"] or None,
            })
        except Exception:
            continue

    return results



def filter_and_rank(
    products: List[Dict[str, Any]],
    max_price: Optional[int],
    size: Optional[str],
    top_n: int = 5,
    query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Client-side safety net: drop anything over budget or outside requested category,
    then sort by price ascending and take top N.
    """
    query_keywords = []
    if query:
        clean_q = re.sub(r"under\s*(?:â‚¹|rs\.?|inr)?\s*\d+", "", query, flags=re.IGNORECASE)
        clean_q = re.sub(r"below\s*(?:â‚¹|rs\.?|inr)?\s*\d+", "", clean_q, flags=re.IGNORECASE)
        clean_q = re.sub(r"size\s*\d+", "", clean_q, flags=re.IGNORECASE)
        words = re.findall(r"[a-zA-Z0-9]+", clean_q.lower())
        stopwords = {
            "a", "an", "the", "and", "or", "in", "of", "for", "to", "is", "it", "that",
            "this", "find", "search", "show", "me", "some", "get", "want", "finally",
            "please", "help", "good", "best", "nice", "buy", "looking", "need"
        }
        query_keywords = [w for w in words if w not in stopwords and len(w) > 1]

    filtered = []
    for p in products:
        name_lower = (p.get("name") or "").lower()

        if max_price is not None and p.get("price_value") is not None:
            if p["price_value"] > max_price:
                continue

        if size:
            p["size_mentioned"] = size in name_lower

        if query_keywords:
            matches_keyword = False
            for kw in query_keywords:
                if kw in name_lower:
                    matches_keyword = True
                    break
                # Category synonym expansion -- maps query keywords to product name signals
                CATEGORY_MAP = {
                    frozenset(["phone", "phones", "mobile", "mobiles", "smartphone", "smartphones"]):
                        ["phone", "mobile", "galaxy", "redmi", "realme", "poco", "oneplus", "smartphone", "5g", "samsung", "iphone", "narzo", "nord"],
                    frozenset(["shoe", "shoes", "sneaker", "sneakers", "footwear", "sport"]):
                        ["shoe", "running", "urbanrun", "sprintline", "trailx", "aerodash", "flexfit", "sneaker", "sandal"],
                    frozenset(["laptop", "laptops", "notebook", "computer", "pc"]):
                        ["laptop", "notebook", "ideapad", "vivobook", "lenovo", "asus", "dell", "hp", "acer"],
                    frozenset(["headphone", "headphones", "earphone", "earphones", "audio"]):
                        ["headphone", "rockerz", "sony", "boat", "jbl", "audio", "wh-1000", "tune", "bass"],
                    frozenset(["earbud", "earbuds", "tws", "airpods", "wireless"]):
                        ["earbud", "tws", "true wireless", "jbl", "boat", "airpod", "buds"],
                    frozenset(["watch", "smartwatch", "smartwatches", "fitness", "tracker"]):
                        ["watch", "smartwatch", "colorfit", "noise", "boat", "wave", "ultima", "tracker"],
                }
                for category_kws, signals in CATEGORY_MAP.items():
                    if kw in category_kws:
                        if any(sig in name_lower for sig in signals):
                            matches_keyword = True
                        break
                if matches_keyword:
                    break
            if not matches_keyword:
                continue

        filtered.append(p)

    filtered.sort(key=lambda p: (p.get("price_value") is None, p.get("price_value") or 0))
    return filtered[:top_n]

