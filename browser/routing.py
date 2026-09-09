"""
routing.py
----------
Category-aware query routing for shopping sites.
Avoids slow sequential cascading through every site by classifying queries
into rough categories and selecting a tailored primary and secondary site.

Target flow:
  Primary site -> Secondary backup site -> Local demo page (safety net)
"""

import re
from typing import Tuple

FASHION_KEYWORDS = {
    "shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "sandals", "heels",
    "shirt", "tshirt", "t-shirt", "dress", "dresses", "jeans", "jacket", "jackets",
    "kurti", "kurtis", "saree", "sarees", "hoodie", "hoodies", "pants", "trousers",
    "top", "tops", "skirt", "suit", "blazer", "ethnic", "wear", "cloth", "clothes",
    "clothing", "sweatshirt", "sweater", "joggers", "trackpants", "kurta", "lehenga",
    # Indic transliteration
    "joote", "joota", "chappal", "kapda", "kapde", "kameez", "dupatta",
}

BEAUTY_KEYWORDS = {
    "makeup", "cosmetics", "lipstick", "lipsticks", "skincare", "skin", "perfume",
    "perfumes", "fragrance", "cologne", "serum", "lotion", "moisturizer", "kajal",
    "foundation", "eyeliner", "mascara", "shampoo", "conditioner", "sunscreen",
    "face wash", "facewash", "nail polish", "lip balm", "concealer", "blush",
    # Indic transliteration
    "attar", "itr", "khushbu", "sugandh", "tel", "ubtan",
}

BUDGET_KEYWORDS = {
    "cheap", "under 500", "under 300", "under 200", "under 100", "affordable",
    "budget", "lowest price", "low price", "cheapest",
    # Indic: sasta, kam daam
    "sasta", "kam daam", "ucit",
}

ELECTRONICS_TECH_KEYWORDS = {
    "laptop", "laptops", "phone", "phones", "mobile", "smartphone", "smartphones",
    "headphones", "earbuds", "earphone", "earphones", "headset", "mouse", "mice",
    "keyboard", "keyboards", "monitor", "charger", "cable", "tablet", "ipad",
    "watch", "smartwatch", "speaker", "speakers", "powerbank", "gadget", "gadgets",
    # Indic transliteration
    "ghadi", "earfon", "mobail", "fon",
}

HOME_APPLIANCE_KEYWORDS = {
    "fan", "pankha", "pankhe", "ceiling fan", "table fan", "exhaust fan",
    "cooler", "ac", "air conditioner", "fridge", "refrigerator", "washing machine",
    "microwave", "mixer", "grinder", "iron", "vacuum cleaner", "heater",
    # Indic transliteration
    "pankha", "pankhe", "farij", "almirah",
}


def _is_budget(q_lower: str) -> bool:
    if re.search(r"\b(cheap|cheapest|affordable|budget|lowest price|low price)\b", q_lower):
        return True
    m = re.search(r"\bunder\s*(?:₹|rs\.?|inr)?\s*(\d+)\b", q_lower)
    if m and int(m.group(1)) <= 500:
        return True
    return False


def get_parallel_sites_for_query(query: str) -> list[str]:
    """
    Returns the list of relevant live shopping sites to search simultaneously in parallel.
    Multi-website comparison ensures top recommendations come from multiple distinct platforms.
    Supports English, Indic transliteration, and multilingual aliases.
    """
    if not query:
        return ["amazon", "snapdeal"]

    q_lower = query.lower()
    # Include both ASCII and Unicode tokens (re.findall with \w+ catches Unicode)
    tokens = set(re.findall(r"[\w]+", q_lower))

    def _matches(kws: set) -> bool:
        for k in kws:
            if " " in k or "-" in k:
                if k in q_lower:
                    return True
            elif k in tokens:
                return True
        return False

    # Explicit store keyword detection if user mentions a specific store
    for store in ("myntra", "nykaa", "meesho"):
        if store in q_lower:
            return [store, "amazon"]

    # 1. Beauty / Cosmetics signals -> Nykaa + Amazon
    if _matches(BEAUTY_KEYWORDS):
        return ["nykaa", "amazon"]

    # 2. Fashion / Apparel signals -> Myntra + Amazon
    if _matches(FASHION_KEYWORDS):
        return ["myntra", "amazon"]

    # 3. Budget signals (< 500) -> Meesho + Snapdeal
    if _is_budget(q_lower):
        return ["meesho", "snapdeal"]

    # 4. Electronics / Tech -> Amazon + Snapdeal
    if _matches(ELECTRONICS_TECH_KEYWORDS):
        return ["amazon", "snapdeal"]

    # 5. Home appliances (fan/pankha, fridge, AC) -> Amazon + Snapdeal
    if _matches(HOME_APPLIANCE_KEYWORDS):
        return ["amazon", "snapdeal"]

    # 6. Default general merchandise -> Amazon + Snapdeal
    return ["amazon", "snapdeal"]


def route_query(query: str) -> Tuple[str, str, str]:
    """
    Classifies a search query and returns (primary_site, secondary_site, tertiary_site).
    Only real, live shopping sites are used:
      - "myntra": Best for fashion, shoes, apparel
      - "nykaa": Best for beauty, cosmetics, fragrances
      - "meesho": Best for budget items under ₹500
      - "amazon": Best for tech, electronics, general catalog
      - "snapdeal": Solid backup for generic/budget merchandise
    """
    sites = get_parallel_sites_for_query(query)
    primary = sites[0] if len(sites) > 0 else "amazon"
    secondary = sites[1] if len(sites) > 1 else "snapdeal"
    tertiary = sites[2] if len(sites) > 2 else "meesho"
    return primary, secondary, tertiary
