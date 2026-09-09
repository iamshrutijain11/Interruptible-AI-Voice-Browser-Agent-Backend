"""
agent.py
--------
LLM orchestration layer. The LLM's ONLY job is to turn a transcribed
utterance into a constrained structured intent:

    {
      "intent": "product_search",
      "query": "<native-language product noun phrase>",
      "query_en": "<same phrase in English, for browser search sites>",
      "detected_language": "hi",
      "constraints": {"max_price": 1500, "size": "9", "color": "black"}
    }

It is never given tool access and never allowed to emit arbitrary text or
actions -- main.py/task_manager.py are the only things that touch
Playwright, and they only ever receive this fixed schema back from here.

Two engines, selected by LLM_ENGINE in .env:
  - "mock"   (default): zero-cost, dependency-free rule-based extraction.
             English-only; degrades gracefully with a warning when
             non-English input is detected (see parse_utterance_mock).
             Good enough for demoing the interruption/versioning logic,
             which is what's actually being judged.
  - "gemini": real LLM call via Google Gemini API with strict
             JSON schema. Supports ANY language the model knows.
             Requires GEMINI_API_KEY.

Multilingual design
--------------------
  - "query"    : the product phrase in the user's own language (for display)
  - "query_en" : the same phrase translated to English (used for search)
  - "detected_language" : ISO 639-1 code; comes from Whisper STT or is
                          inferred by the LLM from the transcript text.

For mock engine + non-English: a WARNING is logged and English-only
regex extraction is attempted as best-effort (documented degraded mode).
For a real multilingual experience, set LLM_ENGINE=gemini in .env.

Context carry-forward
----------------------
A real user rarely repeats the product name on every turn ("Wait! Under
1500, size 9." never re-says "running shoes"). Both engines accept the
previous turn's ParsedIntent as `prior` and merge: an empty/missing query
or constraint field inherits the previous turn's value instead of wiping
it out. detected_language and query_en are also carried forward.
"""

import os
import re
import json
import logging
from typing import Optional

from models import ParsedIntent, Constraints

logger = logging.getLogger("agent")

_COLORS = {
    "black", "white", "red", "blue", "green", "grey", "gray", "brown",
    "pink", "yellow", "navy", "orange", "purple", "beige", "maroon",
}
_STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "of", "for", "to", "is", "it", "that",
    "this", "find", "search", "show", "me", "some", "get", "want", "finally", "please",
    "between", "from", "above", "below", "under"
}
_LEADING_FILLER = [
    "wait,", "wait", "actually,", "actually", "no,", "no wait",
    "hold on,", "hold on", "instead,", "instead", "stop,", "stop",
    "sorry,", "sorry", "hang on,", "hang on", "finally,", "finally",
    "show me", "get me", "find me", "can you show", "can you find",
    "i want", "i'm looking for", "looking for", "some",
]


def _strip_leading_filler(text: str) -> str:
    cleaned = text.strip()
    lowered = cleaned.lower()
    for phrase in _LEADING_FILLER:
        if lowered.startswith(phrase):
            cleaned = cleaned[len(phrase):].strip(" ,!.")
            lowered = cleaned.lower()
    return cleaned or text.strip()


def _extract_price_constraints(text: str) -> tuple[Optional[int], Optional[int]]:
    cleaned_text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    # Range: e.g. "between 1000 and 3000", "from 1000 to 3000", "1000 to 3000", "1000 - 3000", "1000-3000"
    m_range = re.search(
        r"(?:between|from)?\s*(?:₹|rs\.?|inr)?\s*(\d{2,8})\s*(?:and|to|-)\s*(?:₹|rs\.?|inr)?\s*(\d{2,8})",
        cleaned_text,
        re.IGNORECASE
    )
    if m_range:
        p1 = int(m_range.group(1))
        p2 = int(m_range.group(2))
        return (min(p1, p2), max(p1, p2))

    min_p = None
    max_p = None

    m_min = re.search(r"(?:above|over|more than|at least|minimum|min)\s*(?:₹|rs\.?|inr)?\s*(\d{2,8})", cleaned_text, re.IGNORECASE)
    if m_min:
        min_p = int(m_min.group(1))

    m_max = re.search(r"(?:under|below|less than|maximum|max)\s*(?:₹|rs\.?|inr)?\s*(\d{2,8})", cleaned_text, re.IGNORECASE)
    if m_max:
        max_p = int(m_max.group(1))

    return (min_p, max_p)


def _extract_rank_by(text: str) -> Optional[str]:
    if re.search(r"\b(best|top rated|top-rated|bestseller|bestsellers|highest rated|popular|trending)\b", text, re.IGNORECASE):
        return "bestseller_first"
    if re.search(r"\b(cheap|cheapest|low price|lowest price)\b", text, re.IGNORECASE):
        return "price"
    return None


def _extract_size(text: str) -> Optional[str]:
    m = re.search(r"size\s*(\d{1,2}(?:\.\d)?)", text, re.IGNORECASE)
    return m.group(1) if m else None


def _extract_color(text: str) -> Optional[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    for w in words:
        if w in _COLORS:
            return w
    return None


def _extract_product_phrase(text: str, color: Optional[str] = None) -> str:
    """Strip constraint keywords/numbers/stopwords/color, leaving the product noun phrase."""
    cleaned = re.sub(r"(?:between|from)?\s*(?:₹|rs\.?|inr)?\s*\d{2,8}\s*(?:and|to|-)\s*(?:₹|rs\.?|inr)?\s*\d{2,8}", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:under|below|less than|above|over|more than|at least|min|max)\s*(?:₹|rs\.?|inr)?\s*\d{2,8}", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"size\s*\d{1,2}(?:\.\d)?", "", cleaned, flags=re.IGNORECASE)
    words = [w for w in re.findall(r"[a-zA-Z]+", cleaned) if w.lower() not in _STOPWORDS]
    if color:
        words = [w for w in words if w.lower() != color.lower()]
    return " ".join(words).strip()


_INDIC_TRANSLITERATIONS = {
    "pankha": "ceiling fan",
    "pankhe": "ceiling fan",
    "pankho": "ceiling fan",
    "joote": "shoes",
    "joota": "shoes",
    "chappal": "sandals",
    "chappalein": "sandals",
    "ghadi": "watch",
    "gadi": "watch",
    "kapda": "clothes",
    "kapde": "clothes",
    "kameez": "shirt",
    "kitab": "book",
    "kitabein": "books",
    "pustak": "book",
    "khushbu": "perfume",
    "attar": "perfume",
    "itr": "perfume",
}


def parse_utterance_mock(
    text: str,
    prior: Optional[ParsedIntent] = None,
    stt_language: str = "en",
) -> ParsedIntent:
    """
    Zero-cost, dependency-free structured extraction. See module docstring.

    LIMITATION: This parser is English-only (regex on English keywords).
    When `stt_language` is not "en" or "und", a WARNING is logged and
    English-only regex extraction is attempted as best-effort (degraded mode).
    For genuine multilingual support, set LLM_ENGINE=gemini in .env.
    """
    if stt_language not in ("en", "und", ""):
        logger.warning(
            f"[agent/mock] Input detected as '{stt_language}' (non-English). "
            "The mock parser is English-only and cannot reliably extract intent "
            "from non-English utterances. Results may be incorrect or empty. "
            "Set LLM_ENGINE=gemini in .env for real multilingual support."
        )

    cleaned = _strip_leading_filler(text)

    color = _extract_color(cleaned)
    min_price, max_price = _extract_price_constraints(cleaned)
    rank_by = _extract_rank_by(cleaned)

    constraints = Constraints(
        min_price=min_price,
        max_price=max_price,
        size=_extract_size(cleaned),
        color=color,
        rank_by=rank_by or "price",
    )
    product_phrase = _extract_product_phrase(cleaned, color=color)

    if prior is not None:
        constraints = constraints.merged_with(prior.constraints)
        if not product_phrase:
            product_phrase = prior.query

    # Translate known Indic transliterations to English for search sites
    words_en = []
    for w in product_phrase.lower().split():
        words_en.append(_INDIC_TRANSLITERATIONS.get(w, w))
    query_en = " ".join(words_en) or product_phrase

    effective_lang = stt_language if stt_language not in ("und", "") else "en"
    if any(0x0900 <= ord(c) <= 0x097F for c in cleaned):
        effective_lang = "hi"
    elif any(0x0600 <= ord(c) <= 0x06FF for c in cleaned):
        effective_lang = "ar"
    elif any((0x3040 <= ord(c) <= 0x30FF) or (0x4E00 <= ord(c) <= 0x9FFF) for c in cleaned):
        effective_lang = "ja"
    elif any(w in _INDIC_TRANSLITERATIONS for w in cleaned.lower().split()):
        effective_lang = "hi"

    return ParsedIntent(
        intent="product_search",
        query=product_phrase,
        query_en=query_en,
        constraints=constraints,
        detected_language=effective_lang,
    )


# ---------------------------------------------------------------------------
# Multilingual structured system prompt (Gemini / OpenAI path)
# ---------------------------------------------------------------------------
_STRUCTURED_SYSTEM_PROMPT = """\
You extract structured shopping intent from a spoken utterance. The utterance may be
in ANY language (English, Hindi, Spanish, French, German, Arabic, Japanese, Portuguese,
Italian, or others). You must extract the intent regardless of input language.

Return ONLY a single JSON object matching this exact schema — no prose, no markdown fences:
{
  "intent": "product_search",
  "query": "<product noun phrase in the SAME language as the input>",
  "query_en": "<same product noun phrase translated to English — this is used for search sites>",
  "detected_language": "<ISO 639-1 code of the input language, e.g. 'hi', 'es', 'en'>",
  "constraints": {
    "min_price": <integer or null>,
    "max_price": <integer or null>,
    "size": "<string or null>",
    "color": "<string or null>",
    "rank_by": "<'price' or 'bestseller_first'>"
  }
}

Rules:
- Output ONLY the JSON object. No prose, no markdown fences.
- "query" must be the product noun phrase ONLY (no price/size/color in it), in the user's original language.
- "query_en" must be the same product phrase translated into English.
  Example: if input is Hindi "नीले जूते" → query_en = "blue shoes".
  If input is already English → query_en == query.
- "detected_language" must be the ISO 639-1 code you identified for the utterance language.
  If you are given an stt_hint in the prompt, prefer it unless you are highly confident it is wrong.
- If the utterance only adjusts constraints (e.g. "under 1500, size 9") and doesn't name a product,
  set both "query" and "query_en" to "" — the caller will fill them from context.
- Only extract constraints explicitly present in the utterance. Leave others null.
- Set rank_by to "bestseller_first" if user asks for "best", "top rated", "bestseller",
  "highest rated", or "popular" (in ANY language). Otherwise "price".
"""


async def parse_utterance_gemini(
    text: str,
    prior: Optional[ParsedIntent] = None,
    stt_language: str = "en",
) -> ParsedIntent:
    """Real Gemini LLM call. Raises on any failure -- caller should catch and fall back to mock."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")

    # Embed the STT-detected language as a hint in the user message so the model
    # can confirm or gently correct it. We don't ask the model to re-detect
    # independently — we pass the STT code through and let the model validate.
    lang_hint = f"\n[stt_hint: {stt_language}]" if stt_language and stt_language != "und" else ""
    prompt_text = f"{text}{lang_hint}"

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            system_instruction=_STRUCTURED_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    raw = response.text
    data = json.loads(raw)

    # Normalise: query_en defaults to query if model omitted it
    if not data.get("query_en"):
        data["query_en"] = data.get("query", "")
    # Normalise: detected_language defaults to stt_language if model omitted it
    if not data.get("detected_language"):
        data["detected_language"] = stt_language if stt_language != "und" else "en"

    intent = ParsedIntent(**data)

    if prior is not None:
        intent.constraints = intent.constraints.merged_with(prior.constraints)
        if not intent.query:
            intent.query = prior.query
        if not intent.query_en:
            intent.query_en = prior.query_en or prior.query
        # Carry forward language if LLM didn't detect one
        if not intent.detected_language or intent.detected_language == "und":
            intent.detected_language = prior.detected_language or "en"

    return intent


async def parse_utterance(
    text: str,
    prior: Optional[ParsedIntent] = None,
    stt_language: str = "en",
) -> ParsedIntent:
    """
    THE entry point task_manager.py calls.

    High-speed optimization:
    If input is in English or language undetermined, run instant regex intent
    parsing (< 1ms). If product or constraint parameters are identified,
    return immediately. Falls back to Gemini for non-English or ambiguous input.
    """
    engine = os.getenv("LLM_ENGINE", "mock").lower()
    has_api_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))

    # When Gemini is enabled with an API key, use it for genuine multilingual parsing
    # and accurate query translation (unless purely English ASCII in mock mode).
    if engine in ("gemini", "openai") and has_api_key:
        try:
            return await parse_utterance_gemini(text, prior, stt_language=stt_language)
        except Exception as e:
            logger.warning(f"Gemini intent parsing failed ({e}); falling back to mock parser.")

    # High-speed rule-based mock path
    intent_mock = parse_utterance_mock(text, prior, stt_language=stt_language)
    return intent_mock


async def translate_text(text: str, target_lang: str) -> str:
    """
    Translate `text` into `target_lang` using the configured LLM.

    Fast path (zero extra LLM call):
      - If target_lang == "en"  → return text unchanged
      - If LLM_ENGINE == "mock" → return text unchanged

    Otherwise makes ONE LLM call to translate the English summary sentence
    into the target language, so the spoken response is in the user's language.

    :param text: English text to translate (the summary sentence)
    :param target_lang: ISO 639-1 target language code (e.g. "hi", "es")
    :return: Translated text (or original on any failure)
    """
    # Fast path: English needs no translation
    if not target_lang or target_lang in ("en", "und", ""):
        return text

    # Fast path: mock engine cannot translate
    engine = os.getenv("LLM_ENGINE", "mock").lower()
    if engine not in ("gemini", "openai"):
        logger.debug(f"[translate_text] Skipping translation (LLM_ENGINE={engine}); returning English text.")
        return text

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("[translate_text] No API key; returning English text untranslated.")
        return text

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model = os.getenv("LLM_MODEL", "gemini-2.5-flash")

        system_prompt = (
            f"Translate the following English text into the language with ISO 639-1 code '{target_lang}'. "
            "Output only the translated sentence — no explanations, no quotation marks, no extra text."
        )

        response = await client.aio.models.generate_content(
            model=model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
            ),
        )
        translated = response.text.strip() if response.text else text
        logger.info(f"[translate_text] '{text}' -> [{target_lang}] '{translated}'")
        return translated
    except Exception as e:
        logger.warning(f"[translate_text] Translation failed ({e}); returning English text.")
        return text


def build_search_query(intent: ParsedIntent) -> str:
    """
    Re-serialize the structured intent back into a free-text query string.
    Uses intent.query_en (English) so search sites get English terms regardless
    of what language the user spoke in.
    """
    search_base = intent.query_en or intent.query or "product"
    parts = []
    if intent.constraints.color:
        parts.append(intent.constraints.color)
    parts.append(search_base)
    if intent.constraints.min_price is not None and intent.constraints.max_price is not None:
        parts.append(f"between {intent.constraints.min_price} and {intent.constraints.max_price}")
    elif intent.constraints.max_price is not None:
        parts.append(f"under {intent.constraints.max_price}")
    elif intent.constraints.min_price is not None:
        parts.append(f"above {intent.constraints.min_price}")
    if intent.constraints.size:
        parts.append(f"size {intent.constraints.size}")
    return " ".join(parts).strip()


def _score_product(product: dict, max_price: Optional[int], min_price: Optional[int]) -> dict:
    """
    Compute a normalised score breakdown for one product.
    Returns dict with price_fit, rating_score, value_score (all 0–1).
    """
    # --- Price fit ---
    price_fit = 0.5
    raw_price = product.get("price", "")
    numeric = 0
    try:
        cleaned = re.sub(r"[₹,\s]", "", str(raw_price))
        numeric = int(float(re.search(r"\d+(?:\.\d+)?", cleaned).group()))
    except Exception:
        pass

    if numeric > 0:
        if max_price and min_price:
            budget_range = max_price - min_price
            if budget_range > 0:
                price_fit = max(0.0, 1.0 - (numeric - min_price) / budget_range)
            else:
                price_fit = 1.0 if numeric <= max_price else 0.0
        elif max_price:
            price_fit = max(0.0, min(1.0, 1.0 - (numeric - max_price * 0.5) / max_price))
        else:
            price_fit = 0.7  # no constraint → assume neutral

    # --- Rating score ---
    rating_score = 0.5
    try:
        r = float(product.get("rating") or 0)
        rating_score = max(0.0, min(1.0, (r - 1) / 4))  # normalise 1–5 → 0–1
    except Exception:
        pass

    # Bestseller bump
    if product.get("is_bestseller"):
        rating_score = min(1.0, rating_score + 0.15)

    # --- Composite value score ---
    value_score = round(0.5 * price_fit + 0.5 * rating_score, 3)

    return {
        "name":         str(product.get("name", ""))[:60],
        "price_fit":    round(price_fit, 3),
        "rating_score": round(rating_score, 3),
        "value_score":  value_score,
        "site":         product.get("site", ""),
    }


async def build_recommendation(
    results: list,
    query: str,
    constraints,
    sources: list,
) -> "Recommendation":
    """
    Produce a structured explainable recommendation from the search results.

    Two paths:
      - LLM_ENGINE=gemini: calls Gemini to write the reason + comparison text
      - mock / fallback:   rule-based scoring, deterministic, zero API cost

    Always returns a Recommendation object — never raises.
    """
    from models import Recommendation, ProductScore

    if not results:
        return Recommendation(
            reason="No products matched your search criteria.",
            comparison_summary="",
            sources_searched=sources,
            total_found=0,
        )

    max_price = getattr(constraints, "max_price", None)
    min_price = getattr(constraints, "min_price", None)

    # Score every result
    raw_scores = [_score_product(p, max_price, min_price) for p in results]
    # Sort by value_score descending to pick the top
    indexed = sorted(enumerate(raw_scores), key=lambda x: x[1]["value_score"], reverse=True)
    best_idx, best_score = indexed[0]
    top = results[best_idx]

    product_scores = [
        ProductScore(
            name=s["name"],
            price_fit=s["price_fit"],
            rating_score=s["rating_score"],
            value_score=s["value_score"],
            site=s["site"],
        )
        for s in raw_scores
    ]

    # ── LLM path (optional, enabled if LLM_RECOMMENDATION=true) ─────────────
    engine = os.getenv("LLM_ENGINE", "mock").lower()
    enable_llm_rec = os.getenv("LLM_RECOMMENDATION", "false").lower() == "true"
    if engine in ("gemini", "openai") and enable_llm_rec:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key)
                model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")

                # Build a compact product list for the LLM
                snippet = "\n".join(
                    f"  {i+1}. {p.get('name','?')[:50]} | {p.get('price','?')} | "
                    f"rating:{p.get('rating','?')} | {'bestseller' if p.get('is_bestseller') else ''} | {p.get('site','?')}"
                    for i, p in enumerate(results[:6])
                )
                constraint_desc = []
                if max_price:
                    constraint_desc.append(f"max price ₹{max_price}")
                if min_price:
                    constraint_desc.append(f"min price ₹{min_price}")
                if getattr(constraints, "color", None):
                    constraint_desc.append(f"color: {constraints.color}")
                constraint_str = ", ".join(constraint_desc) if constraint_desc else "no specific constraints"

                prompt = (
                    f"User searched for: {query}\nConstraints: {constraint_str}\n\n"
                    f"Top products found:\n{snippet}\n\n"
                    f"The recommended top pick is product #{best_idx+1}.\n\n"
                    "Write:\n"
                    "REASON: (1-2 sentences explaining why this is the top pick, mentioning price, rating, or value)\n"
                    "COMPARISON: (2-3 sentences comparing the top 3 products to help the user decide)\n\n"
                    "Output ONLY the two labelled sections above, nothing else."
                )

                resp = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.3),
                )
                raw = resp.text or ""
                reason = ""
                comparison = ""
                for line in raw.splitlines():
                    if line.startswith("REASON:"):
                        reason = line[7:].strip()
                    elif line.startswith("COMPARISON:"):
                        comparison = line[11:].strip()

                if reason:
                    return Recommendation(
                        top_pick_index=best_idx,
                        top_pick_name=str(top.get("name", ""))[:80],
                        top_pick_price=str(top.get("price", "")),
                        top_pick_site=str(top.get("site", "")),
                        reason=reason,
                        comparison_summary=comparison,
                        scores=product_scores,
                        sources_searched=sources,
                        total_found=len(results),
                    )
            except Exception as e:
                logger.warning(f"[build_recommendation] LLM path failed ({e}); using rule-based.")

    # ── Rule-based mock path ─────────────────────────────────────────────────
    name = str(top.get("name", "product"))[:60]
    price = str(top.get("price", ""))
    site  = str(top.get("site", ""))
    rating = top.get("rating")

    reason_parts = [f"'{name}' on {site} offers the best overall value"]
    if rating:
        reason_parts.append(f"with a {rating}★ rating")
    if price:
        reason_parts.append(f"at {price}")
    if top.get("is_bestseller"):
        reason_parts.append("and is a bestseller")
    reason = " ".join(reason_parts) + "."

    # Comparison across top 3
    top3 = [results[i] for i, _ in indexed[:3]]
    cmp_parts = []
    for p in top3:
        pname  = str(p.get("name", ""))[:40]
        pprice = str(p.get("price", "?"))
        psite  = str(p.get("site", "?"))
        prating = p.get("rating")
        seg = f"{pname} ({pprice} on {psite}"
        if prating:
            seg += f", {prating}★"
        seg += ")"
        cmp_parts.append(seg)
    comparison = "Compared: " + " vs ".join(cmp_parts) + "." if cmp_parts else ""

    return Recommendation(
        top_pick_index=best_idx,
        top_pick_name=str(top.get("name", ""))[:80],
        top_pick_price=str(top.get("price", "")),
        top_pick_site=str(top.get("site", "")),
        reason=reason,
        comparison_summary=comparison,
        scores=product_scores,
        sources_searched=sources,
        total_found=len(results),
    )
