"""
models.py
---------
Shared data shapes for the whole backend. Nothing here talks to Playwright,
Whisper, Rime, or an LLM API directly -- these are the plain data contracts
that task_manager.py, agent.py, events.py, and main.py all pass around.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import time


class TaskState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    BROWSING = "BROWSING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class Constraints(BaseModel):
    """Structured constraints the LLM (or the mock parser) extracts from speech."""
    max_price: Optional[int] = None
    size: Optional[str] = None
    color: Optional[str] = None

    def merged_with(self, previous: "Constraints") -> "Constraints":
        """New values win; anything unset here falls back to the previous turn's value."""
        return Constraints(
            max_price=self.max_price if self.max_price is not None else previous.max_price,
            size=self.size if self.size is not None else previous.size,
            color=self.color if self.color is not None else previous.color,
        )


class ParsedIntent(BaseModel):
    """
    The LLM's (constrained) structured output. This is the ONLY thing the
    LLM is allowed to produce -- it is never allowed to directly call
    browser actions or return arbitrary text.
    """
    intent: str = "product_search"
    query: str = ""          # the product phrase, e.g. "black running shoes"
    constraints: Constraints = Field(default_factory=Constraints)


class Task(BaseModel):
    task_id: str
    query: str
    constraints: Constraints
    state: TaskState = TaskState.THINKING
    results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
