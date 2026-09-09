"""
agent.py
--------
LLM orchestration layer. The LLM's ONLY job is to turn a transcribed
utterance into a constrained structured intent:

    {
      "intent": "product_search",
      "query": "black running shoes",
      "constraints": {"max_price": 1500, "size": "9", "color": "black"}
    }

It is never given tool access and never allowed to emit arbitrary text or
actions -- main.py/task_manager.py are the only things that touch
Playwright, and they only ever receive this fixed schema back from here.

Two engines, selected by LLM_ENGINE in .env:
  - "mock"   (default): zero-cost, dependency-free rule-based extraction.
             Good enough for demoing the interruption/versioning logic,
             which is what's actually being judged.
  - "gemini": real LLM call via Google Gemini API with strict
             JSON schema. Requires GEMINI_API_KEY.

Context carry-forward
----------------------
A real user rarely repeats the product name on every turn ("Wait! Under
1500, size 9." never re-says "running shoes"). Both engines accept the
previous turn's ParsedIntent as `prior` and merge: an empty/missing query
or constraint field inherits the previous turn's value instead of wiping
it out.
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
    "this", "find", "search", "show", "me", "some", "get", "want", "finally", "please"
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


def _extract_max_price(text: str) -> Optional[int]:
    cleaned_text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    m = re.search(r"under\s*(?:₹|rs\.?|inr)?\s*(\d{2,8})", cleaned_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"below\s*(?:₹|rs\.?|inr)?\s*(\d{2,8})", cleaned_text, re.IGNORECASE)
    return int(m.group(1)) if m else None


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
    cleaned = re.sub(r"under\s*(?:₹|rs\.?|inr)?\s*\d{2,6}", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"below\s*(?:₹|rs\.?|inr)?\s*\d{2,6}", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"size\s*\d{1,2}(?:\.\d)?", "", cleaned, flags=re.IGNORECASE)
    words = [w for w in re.findall(r"[a-zA-Z]+", cleaned) if w.lower() not in _STOPWORDS]
    if color:
        words = [w for w in words if w.lower() != color.lower()]
    return " ".join(words).strip()


def parse_utterance_mock(text: str, prior: Optional[ParsedIntent] = None) -> ParsedIntent:
    """Zero-cost, dependency-free structured extraction. See module docstring."""
    cleaned = _strip_leading_filler(text)

    color = _extract_color(cleaned)
    constraints = Constraints(
        max_price=_extract_max_price(cleaned),
        size=_extract_size(cleaned),
        color=color,
    )
    product_phrase = _extract_product_phrase(cleaned, color=color)

    if prior is not None:
        constraints = constraints.merged_with(prior.constraints)
        if not product_phrase:
            product_phrase = prior.query

    return ParsedIntent(intent="product_search", query=product_phrase, constraints=constraints)


_GEMINI_SYSTEM_PROMPT = """You convert a spoken shopping instruction into STRICT JSON matching this schema, nothing else:
{"intent": "product_search", "query": "<product noun phrase, no constraints in it>", "constraints": {"max_price": <int or null>, "size": "<string or null>", "color": "<string or null>"}}

Rules:
- Output ONLY the JSON object. No prose, no markdown fences.
- If the utterance only adjusts constraints (e.g. "under 1500, size 9") and doesn't name a product, set "query" to an empty string "" -- the caller will fill it in from context, do NOT guess a product.
- Only extract constraints explicitly present in the utterance. Leave others null.
"""


async def parse_utterance_gemini(text: str, prior: Optional[ParsedIntent] = None) -> ParsedIntent:
    """Real Gemini LLM call. Raises on any failure -- caller should catch and fall back to mock."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")

    response = await client.aio.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=_GEMINI_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    raw = response.text
    data = json.loads(raw)
    intent = ParsedIntent(**data)

    if prior is not None:
        intent.constraints = intent.constraints.merged_with(prior.constraints)
        if not intent.query:
            intent.query = prior.query

    return intent


async def parse_utterance(text: str, prior: Optional[ParsedIntent] = None) -> ParsedIntent:
    """
    THE entry point task_manager.py calls. Picks the engine from LLM_ENGINE
    (.env), and transparently falls back to the mock parser if the real LLM
    call fails for any reason (missing key, network, rate limit) -- a
    hackathon demo should never hard-fail because an LLM API hiccupped.
    """
    engine = os.getenv("LLM_ENGINE", "mock").lower()

    if engine in ("gemini", "openai"):
        try:
            return await parse_utterance_gemini(text, prior)
        except Exception as e:
            logger.warning(f"Gemini intent parsing failed ({e}); falling back to mock parser.")

    return parse_utterance_mock(text, prior)


def build_search_query(intent: ParsedIntent) -> str:
    """
    Re-serialize the structured intent back into a free-text query string,
    since Person 2's search_products(query, task_id) takes plain text (and
    internally re-derives max_price/size from it -- see browser/extraction.py).
    This is the bridge between the LLM's structured world and the browser
    layer's plain-text interface, without needing to touch browser/ at all.
    """
    parts = []
    if intent.constraints.color:
        parts.append(intent.constraints.color)
    parts.append(intent.query or "product")
    if intent.constraints.max_price is not None:
        parts.append(f"under {intent.constraints.max_price}")
    if intent.constraints.size:
        parts.append(f"size {intent.constraints.size}")
    return " ".join(parts).strip()
