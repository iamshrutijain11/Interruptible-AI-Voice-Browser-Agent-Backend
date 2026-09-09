"""
models.py
---------
Shared data shapes for the whole backend.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import time


class TaskState(str, Enum):
    IDLE               = "IDLE"
    LISTENING          = "LISTENING"
    THINKING           = "THINKING"
    PLANNING           = "PLANNING"
    BROWSING           = "BROWSING"
    ANALYZING          = "ANALYZING"
    RECOMMENDING       = "RECOMMENDING"
    AWAITING_APPROVAL  = "AWAITING_APPROVAL"
    SPEAKING           = "SPEAKING"
    INTERRUPTED        = "INTERRUPTED"
    COMPLETED          = "COMPLETED"
    ERROR              = "ERROR"


class StepStatus(str, Enum):
    PENDING     = "pending"
    RUNNING     = "running"
    COMPLETED   = "completed"
    FAILED      = "failed"
    INTERRUPTED = "interrupted"
    SKIPPED     = "skipped"


class PlanStep(BaseModel):
    """One step in the agent's visible task plan."""
    id: str
    label: str
    status: StepStatus = StepStatus.PENDING
    detail: Optional[str] = None


# The 8 canonical plan steps, in execution order.
# Instantiated fresh for each task by task_manager.py.
PLAN_TEMPLATE: List[Dict[str, str]] = [
    {"id": "understand",  "label": "Understanding intent"},
    {"id": "plan",        "label": "Planning search strategy"},
    {"id": "browse",      "label": "Browsing websites"},
    {"id": "extract",     "label": "Extracting products"},
    {"id": "analyze",     "label": "Analyzing & comparing"},
    {"id": "recommend",   "label": "Explaining recommendation"},
    {"id": "approve",     "label": "Awaiting your approval"},
    {"id": "act",         "label": "Executing action"},
]


class ProductScore(BaseModel):
    """Per-product scoring breakdown for the recommendation panel."""
    name: str
    price_fit:    float = 0.0   # 0–1: how well price fits constraints
    rating_score: float = 0.0   # 0–1: normalised rating
    value_score:  float = 0.0   # 0–1: composite value score
    site: Optional[str] = None


class Recommendation(BaseModel):
    """
    Structured explainable recommendation produced by build_recommendation().
    Consumed by RecommendationPanel on the frontend.
    """
    top_pick_index: int = 0          # index into the results list
    top_pick_name:  str = ""
    top_pick_price: str = ""
    top_pick_site:  str = ""
    reason: str = ""                 # 1–2 sentence human-readable justification
    comparison_summary: str = ""     # 2–3 sentence cross-product comparison
    scores: List[ProductScore] = Field(default_factory=list)
    sources_searched: List[str] = Field(default_factory=list)
    total_found: int = 0


class Constraints(BaseModel):
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    size:      Optional[str] = None
    color:     Optional[str] = None
    rank_by:   Optional[str] = "price"

    def merged_with(self, previous: "Constraints") -> "Constraints":
        return Constraints(
            min_price=self.min_price if self.min_price is not None else previous.min_price,
            max_price=self.max_price if self.max_price is not None else previous.max_price,
            size=self.size   if self.size   is not None else previous.size,
            color=self.color if self.color  is not None else previous.color,
            rank_by=self.rank_by if (self.rank_by and self.rank_by != "price")
                    else (previous.rank_by or "price"),
        )


class ParsedIntent(BaseModel):
    """
    Structured LLM output. Multilingual fields:
      detected_language — ISO 639-1 code
      query_en          — English translation for search sites
    """
    intent: str = "product_search"
    query:    str = ""
    query_en: str = ""
    constraints: Constraints = Field(default_factory=Constraints)
    detected_language: str = "en"


class Task(BaseModel):
    task_id:     str
    query:       str
    constraints: Constraints
    state:       TaskState = TaskState.THINKING
    plan:        List[PlanStep] = Field(default_factory=list)
    results:     List[Dict[str, Any]] = Field(default_factory=list)
    recommendation: Optional[Dict[str, Any]] = None
    error:       Optional[str] = None
    created_at:  float = Field(default_factory=time.time)
