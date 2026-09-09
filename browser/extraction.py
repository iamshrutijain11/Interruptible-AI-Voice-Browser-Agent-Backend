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



def parse_price_range(query: str) -> tuple[Optional[int], Optional[int]]:
    text = re.sub(r"(\d+),(\d+)", r"\1\2", query.lower())
    m_range = re.search(
        r"(?:between|from)?\s*(?:₹|rs\.?|inr)?\s*(\d{2,8})\s*(?:and|to|-)\s*(?:₹|rs\.?|inr)?\s*(\d{2,8})",
        text,
        re.IGNORECASE,
    )
    if m_range:
        p1 = int(m_range.group(1))
        p2 = int(m_range.group(2))
        return (min(p1, p2), max(p1, p2))

    min_p = None
    max_p = None

    m_min = re.search(r"(?:above|over|more than|at least|minimum|min)\s*(?:₹|rs\.?|inr)?\s*(\d{2,8})", text, re.IGNORECASE)
    if m_min:
        min_p = int(m_min.group(1))

    m_max = re.search(r"(?:under|below|less than|maximum|max)\s*(?:₹|rs\.?|inr)?\s*(\d{2,8})", text, re.IGNORECASE)
    if not m_max:
        m_max = re.search(r"(?:₹|rs\.?|inr)\s*(\d{2,8})\s*(?:or less|max)", text, re.IGNORECASE)
    if m_max:
        max_p = int(m_max.group(1))

    return (min_p, max_p)


def parse_max_price(query: str) -> Optional[int]:
    _, max_p = parse_price_range(query)
    return max_p


def parse_min_price(query: str) -> Optional[int]:
    min_p, _ = parse_price_range(query)
    return min_p


def parse_size(query: str) -> Optional[str]:
    """Extract a shoe size like 'size 9' -> '9'."""
    m = re.search(r"size\s*(\d{1,2}(?:\.\d)?)", query.lower())
    return m.group(1) if m else None


def price_to_int(price_text: str) -> Optional[int]:
    """'₹1,299' -> 1299. Returns None if unparseable."""
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
    max_cards: int = 15,
) -> List[Dict[str, Any]]:
    """
    Generic card-based extractor using a single page.evaluate() call for speed.
    Extracts name, price, link, highest-resolution image, rating, review count,
    and bestseller status in a single pass.
    """
    try:
        raw = await page.evaluate(
            """([cardSel, nameSel, priceSel, linkSel, imageSel, maxCards, pageUrl]) => {
                const cards = Array.from(document.querySelectorAll(cardSel)).slice(0, maxCards);
                let origin = '';
                try { origin = new URL(pageUrl).origin; } catch(e) {}

                function getBestImage(card, imageSel) {
                    let bestUrl = '';
                    let maxScore = -1;
                    let imgEls = [];
                    if (imageSel) {
                        for (const s of imageSel.split(',').map(x => x.trim())) {
                            if (s) imgEls.push(...card.querySelectorAll(s));
                        }
                    }
                    if (imgEls.length === 0) {
                        imgEls = Array.from(card.querySelectorAll('img'));
                    }
                    for (const el of imgEls) {
                        if (!el) continue;
                        const srcset = el.getAttribute('srcset') || el.getAttribute('data-srcset') || '';
                        if (srcset) {
                            const parts = srcset.split(',');
                            for (const part of parts) {
                                const tokens = part.trim().split(/\\s+/);
                                const url = tokens[0];
                                if (!url || url.startsWith('data:') || url.includes('placeholder') || url.includes('1x1') || url.includes('pixel.gif')) continue;
                                let score = 1;
                                if (tokens[1]) {
                                    const num = parseFloat(tokens[1]);
                                    if (!isNaN(num)) score = tokens[1].endsWith('w') ? num : num * 300;
                                }
                                if (score > maxScore) {
                                    maxScore = score;
                                    bestUrl = url;
                                }
                            }
                        }
                        const attrs = ['data-src', 'data-original', 'data-lazy-src', 'src'];
                        for (const attr of attrs) {
                            const val = el.getAttribute(attr);
                            if (val && !val.startsWith('data:') && !val.includes('placeholder') && !val.includes('1x1') && !val.includes('pixel.gif')) {
                                if (maxScore < 0) {
                                    maxScore = 0;
                                    bestUrl = val;
                                }
                                break;
                            }
                        }
                        if (bestUrl && maxScore > 0) break;
                    }
                    if (bestUrl && bestUrl.startsWith('//')) bestUrl = 'https:' + bestUrl;
                    return bestUrl;
                }

                return cards.map(card => {
                    try {
                        let name = '';
                        // 1. Myntra multi-element title (brand + product description)
                        const brandEl = card.querySelector('.product-brand, h3.product-brand');
                        const prodEl = card.querySelector('.product-product, h4.product-product');
                        if (brandEl && prodEl) {
                            name = (brandEl.textContent.trim() + ' ' + prodEl.textContent.trim()).trim();
                        }

                        // 2. Try configured name selector
                        if (!name) {
                            for (const ns of nameSel.split(',').map(s => s.trim())) {
                                if (!ns) continue;
                                try {
                                    const el = card.querySelector(ns);
                                    if (el && el.textContent.trim()) { name = el.textContent.trim(); break; }
                                } catch(e) {}
                            }
                        }

                        // 3. For Amazon / standard sites, h2 or h2 a often contains the complete product name
                        const h2El = card.querySelector('h2 a, h2');
                        if (h2El && h2El.textContent.trim()) {
                            const fullH2 = h2El.textContent.replace(/\\s+/g, ' ').trim();
                            if (!name || fullH2.length > name.length) {
                                name = fullH2;
                            }
                        }

                        // 4. If name is just brand (<= 2 words), check for description link / recipe
                        if (name && name.split(/\\s+/).length <= 2) {
                            const descEl = card.querySelector('a.a-text-normal, [data-cy="title-recipe"] a, h2 + a');
                            if (descEl && descEl.textContent.trim()) {
                                const desc = descEl.textContent.replace(/\\s+/g, ' ').trim();
                                if (desc.length > name.length && !desc.toLowerCase().includes(name.toLowerCase())) {
                                    name = name + ' ' + desc;
                                } else if (desc.length > name.length) {
                                    name = desc;
                                }
                            }
                        }

                        if (!name) {
                            const fallbackTitle = card.querySelector('h2, h3, h4, [class*="title"], [class*="brand"]');
                            if (fallbackTitle && fallbackTitle.textContent.trim()) {
                                name = fallbackTitle.textContent.trim();
                            }
                        }

                        let price = '';
                        for (const ps of priceSel.split(',').map(s => s.trim())) {
                            if (!ps) continue;
                            try {
                                const el = card.querySelector(ps);
                                if (el && el.textContent.trim()) { price = el.textContent.trim(); break; }
                            } catch(e) {}
                        }
                        if (!price) {
                            const spans = Array.from(card.querySelectorAll('span, div, p, h5, b, strong'));
                            for (const s of spans) {
                                const t = s.textContent.trim();
                                if (/^(?:Price\\s*)?[₹|Rs\\.?]\\s*[\\d,]+/i.test(t)) {
                                    price = t;
                                    break;
                                }
                            }
                        }
                        if (!name || !price) return null;

                        let href = '';
                        let hrefRaw = '';
                        if (card.tagName === 'A' && card.getAttribute('href')) {
                            hrefRaw = card.getAttribute('href');
                            href = hrefRaw;
                        }
                        if (!href) {
                            for (const ls of linkSel.split(',').map(s => s.trim())) {
                                const el = card.querySelector(ls);
                                if (el && el.getAttribute('href')) {
                                    hrefRaw = el.getAttribute('href');
                                    href = hrefRaw;
                                    break;
                                }
                            }
                        }
                        if (!href) {
                            const h2a = card.querySelector('h2 a, a:has(h2)');
                            if (h2a && h2a.getAttribute('href')) href = h2a.getAttribute('href');
                        }
                        if (!href) {
                            const anyProd = card.querySelector("a[href*='/dp/'], a[href*='/gp/'], a[href*='/p/'], a.a-link-normal[href]");
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

                        const image = getBestImage(card, imageSel);

                        // Bestseller detection
                        const cardText = card.textContent || '';
                        let isBestseller = false;
                        if (/best\\s*seller|#1\\s*best\\s*seller|top\\s*rated|trending/i.test(cardText)) {
                            const badgeEl = card.querySelector('.a-badge-label, .a-badge-text, [class*="badge"], [class*="bestseller"], [class*="rating-tag"]');
                            if (badgeEl && /best\\s*seller|top\\s*rated|trending|amazon\\'?s\\s*choice/i.test(badgeEl.textContent)) {
                                isBestseller = true;
                            } else if (/\\b(best\\s*seller|bestseller)\\b/i.test(cardText)) {
                                isBestseller = true;
                            }
                        }

                        // Rating detection
                        let rating = null;
                        const ratingEl = card.querySelector('span.a-icon-alt, [aria-label*="out of 5 stars"], [aria-label*="stars"], [class*="rating"], [class*="ratingContainer"], [class*="rating-score"]');
                        if (ratingEl) {
                            const rText = ratingEl.getAttribute('aria-label') || ratingEl.textContent || '';
                            const m = rText.match(/(\\d+(?:\\.\\d+)?)\\s*(?:out of 5|\\/5|\\/ 5|stars)?/i);
                            if (m) {
                                const val = parseFloat(m[1]);
                                if (val >= 1 && val <= 5) rating = val;
                            }
                        }
                        if (!rating) {
                            const m = cardText.match(/\\b([1-5]\\.[0-9])\\s*(?:★|⭐|\\/5)?\\b/);
                            if (m) rating = parseFloat(m[1]);
                        }

                        // Review count detection
                        let reviewCount = null;
                        const revEl = card.querySelector('span.a-size-base.s-underline-text, [class*="review"], [class*="ratingCount"]');
                        if (revEl) {
                            const revText = revEl.textContent.replace(/,/g, '').trim();
                            const m = revText.match(/(\\d+(?:\\.\\d+)?)\\s*(k|k reviews|reviews|\\))/i);
                            if (m) {
                                let cnt = parseFloat(m[1]);
                                if (/k/i.test(m[2])) cnt *= 1000;
                                reviewCount = Math.round(cnt);
                            } else {
                                const d = revText.match(/\\d+/);
                                if (d) reviewCount = parseInt(d[0]);
                            }
                        }
                        if (reviewCount === null) {
                            const m = cardText.match(/\\((\\s*[\\d,]+\\s*)\\)|\\b([\\d,]+)\\s+ratings\\b|\\b([\\d,]+)\\s+reviews\\b/i);
                            if (m) {
                                const numStr = (m[1] || m[2] || m[3]).replace(/,/g, '').trim();
                                const cnt = parseInt(numStr);
                                if (!isNaN(cnt) && cnt < 1000000) reviewCount = cnt;
                            }
                        }

                        return { name, price, href, hrefRaw, image, rating, reviewCount, isBestseller };
                    } catch(e) { return null; }
                }).filter(x => x !== null);
            }""",
            [card_selector, name_selector, price_selector, link_selector,
             image_selector, max_cards, page.url],
        )
    except Exception as e:
        logger.warning("[extract] JS evaluation failed: %s", e)
        return []

    results: List[Dict[str, Any]] = []
    for i, item in enumerate(raw or []):
        try:
            if i < 3:
                logger.info(
                    "[href-debug] card[%d] linkSel=%r hrefRaw=%r -> href=%r rating=%s bs=%s",
                    i, link_selector, item.get("hrefRaw", ""), item.get("href", ""),
                    item.get("rating"), item.get("isBestseller")
                )
            results.append({
                "name":          item["name"],
                "price":         item["price"],
                "price_value":   price_to_int(item["price"]),
                "url":           item["href"] or None,
                "image":         item["image"] or None,
                "rating":        item.get("rating"),
                "review_count":  item.get("reviewCount"),
                "is_bestseller": bool(item.get("isBestseller", False)),
            })
        except Exception:
            continue

    return results



def filter_and_rank(
    products: List[Dict[str, Any]],
    max_price: Optional[int] = None,
    size: Optional[str] = None,
    top_n: int = 5,
    query: Optional[str] = None,
    min_price: Optional[int] = None,
    rank_by: str = "price",
) -> List[Dict[str, Any]]:
    """
    Client-side safety net: drop anything outside requested price range or category,
    then sort by ranking criteria (price or bestseller_first) and take top N.
    """
    query_keywords = []
    if query:
        clean_q = re.sub(r"(?:between|from)?\s*(?:₹|rs\.?|inr)?\s*\d+\s*(?:and|to|-)\s*(?:₹|rs\.?|inr)?\s*\d+", "", query, flags=re.IGNORECASE)
        clean_q = re.sub(r"(?:under|below|less than|above|over|more than|at least|min|max)\s*(?:₹|rs\.?|inr)?\s*\d+", "", clean_q, flags=re.IGNORECASE)
        clean_q = re.sub(r"size\s*\d+", "", clean_q, flags=re.IGNORECASE)
        words = re.findall(r"[a-zA-Z0-9]+", clean_q.lower())
        stopwords = {
            "a", "an", "the", "and", "or", "in", "of", "for", "to", "is", "it", "that",
            "this", "find", "search", "show", "me", "some", "get", "want", "finally",
            "please", "help", "good", "best", "nice", "buy", "looking", "need", "between", "from"
        }
        query_keywords = [w for w in words if w not in stopwords and len(w) > 1]

    price_filtered = []
    filtered = []
    for p in products:
        name_lower = (p.get("name") or "").lower()

        if max_price is not None and p.get("price_value") is not None:
            if p["price_value"] > max_price:
                continue

        if min_price is not None and p.get("price_value") is not None:
            if p["price_value"] < min_price:
                continue

        if size:
            p["size_mentioned"] = size in name_lower

        price_filtered.append(p)

        if query_keywords:
            matches_keyword = False
            for kw in query_keywords:
                if kw in name_lower:
                    matches_keyword = True
                    break
                # Category synonym expansion
                CATEGORY_MAP = {
                    frozenset(["phone", "phones", "mobile", "mobiles", "smartphone", "smartphones"]):
                        ["phone", "mobile", "galaxy", "redmi", "realme", "poco", "oneplus", "smartphone", "5g", "samsung", "iphone", "narzo", "nord"],
                    frozenset(["shoe", "shoes", "sneaker", "sneakers", "footwear", "sport", "running"]):
                        ["shoe", "running", "urbanrun", "sprintline", "trailx", "aerodash", "flexfit", "sneaker", "sandal", "casual", "walk", "cushion", "footwear", "trainer", "trainers", "jog", "jogging", "gym", "mesh", "slip-on"],
                    frozenset(["laptop", "laptops", "notebook", "computer", "pc"]):
                        ["laptop", "notebook", "ideapad", "vivobook", "lenovo", "asus", "dell", "hp", "acer"],
                    frozenset(["headphone", "headphones", "earphone", "earphones", "audio"]):
                        ["headphone", "rockerz", "sony", "boat", "jbl", "audio", "wh-1000", "tune", "bass"],
                    frozenset(["earbud", "earbuds", "tws", "airpods", "wireless"]):
                        ["earbud", "tws", "true wireless", "jbl", "boat", "airpod", "buds"],
                    frozenset(["watch", "smartwatch", "smartwatches", "fitness", "tracker"]):
                        ["watch", "smartwatch", "colorfit", "noise", "boat", "wave", "ultima", "tracker"],
                    frozenset(["lipstick", "lipsticks", "cosmetics", "lip"]):
                        ["lipstick", "matte", "gloss", "lip", "lakme", "maybelline", "sugar", "shade"],
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

    # Fallback to price_filtered if keyword filter was too strict for real site results
    if not filtered and price_filtered:
        filtered = price_filtered

    if rank_by == "bestseller_first":
        filtered.sort(
            key=lambda p: (
                0 if p.get("is_bestseller") else 1,
                -(p.get("rating") or 0.0),
                -(p.get("review_count") or 0),
                p.get("price_value") if p.get("price_value") is not None else 999999,
            )
        )
    else:
        filtered.sort(key=lambda p: (p.get("price_value") is None, p.get("price_value") or 0))

    return filtered[:top_n]

