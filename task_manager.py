"""
task_manager.py
----------------
THE most important file in the project, per the spec: central task
versioning. Exactly one task is ever "current". Every asynchronous
operation below carries its task_id, and before accepting/broadcasting any
result, we check `task_id == self.current_task_id` -- if not, the result is
silently discarded. This is what makes stale results from an old,
interrupted task impossible to surface, no matter how the timing lands.

This file is the ONLY place that calls into both Person 1's voice module
and Person 2's browser module in the same breath -- neither of those
modules knows the other exists, and neither is modified here.

Full pipeline per utterance:
    text (from STT)
      -> agent.parse_utterance()          structured intent (LLM or mock)
      -> [interrupt previous task if one exists]
      -> register new task, broadcast task.started
      -> speak acknowledgement            (Person 1: rime_service.speak)
      -> browser.search_products()        (Person 2)
      -> [DISCARD if no longer current]
      -> broadcast browser.result
      -> speak result summary             (Person 1)
      -> broadcast task.completed
"""

import uuid
import asyncio
import logging
from typing import Optional, Dict, Callable, Awaitable, Any

import events
import agent
from models import Task, TaskState, ParsedIntent

from voice import voice_manager, rime_service
from browser import (
    search_products,
    create_task as browser_create_task,
    cancel_task as browser_cancel_task,
)

logger = logging.getLogger("task_manager")

BroadcastFn = Callable[[dict], Awaitable[None]]


class TaskManager:
    def __init__(self):
        self.current_task_id: Optional[str] = None
        self.tasks: Dict[str, Task] = {}
        self._last_intent: Optional[ParsedIntent] = None

    # ---------------------------------------------------------------- core
    def is_current(self, task_id: str) -> bool:
        """THE check. Every result/state update goes through this first."""
        return task_id is not None and task_id == self.current_task_id

    def _new_task_id(self) -> str:
        return f"task_{uuid.uuid4().hex[:8]}"

    async def _emit_state(self, task_id: str, state: TaskState, broadcast: BroadcastFn) -> None:
        if not self.is_current(task_id):
            return
        self.tasks[task_id].state = state
        await broadcast(events.state_changed(task_id, state.value))

    # ------------------------------------------------------------ pipeline
    async def handle_utterance(self, text: str, broadcast: BroadcastFn) -> Optional[dict]:
        """
        THE entry point. Called with whatever Person 1's STT produced.
        Returns the final result dict, or None if the task went stale
        before completion (discarded per the spec's core rule).
        """
        old_task_id = self.current_task_id

        await broadcast(events.transcript_updated(old_task_id, text))

        intent = await agent.parse_utterance(text, self._last_intent)
        self._last_intent = intent
        search_query = agent.build_search_query(intent)

        new_task_id = self._new_task_id()

        if old_task_id:
            await self._interrupt(old_task_id, new_task_id, broadcast)

        self.tasks[new_task_id] = Task(
            task_id=new_task_id,
            query=search_query,
            constraints=intent.constraints,
            state=TaskState.THINKING,
        )
        self.current_task_id = new_task_id
        await broadcast(events.task_started(new_task_id, search_query, intent.constraints.dict()))

        # --- Speak an acknowledgement (Person 1) while browsing starts concurrently ---
        await self._emit_state(new_task_id, TaskState.SPEAKING, broadcast)
        ack_text = f"Searching for {search_query}..."
        await broadcast(events.speech_started(new_task_id, ack_text))
        asyncio.create_task(asyncio.to_thread(rime_service.speak, ack_text, new_task_id))

        # --- Browse (Person 2) ---
        await self._emit_state(new_task_id, TaskState.BROWSING, broadcast)
        await broadcast(events.browser_started(new_task_id))

        browser_create_task(new_task_id, search_query)
        result = await search_products(search_query, new_task_id)

        # === THE core versioning check: discard if superseded while browsing ===
        if not self.is_current(new_task_id) or result.get("status") == "stale":
            logger.info(f"Task '{new_task_id}' went stale before its result could be delivered. Discarding.")
            return None

        if result.get("status") not in ("completed", "completed_empty"):
            error_msg = result.get("error", "Unknown error during search")
            self.tasks[new_task_id].error = error_msg
            await self._emit_state(new_task_id, TaskState.ERROR, broadcast)
            await broadcast(events.task_error(new_task_id, error_msg))
            asyncio.create_task(asyncio.to_thread(rime_service.speak, "Sorry, I ran into a problem while searching.", new_task_id))
            return result

        results = result.get("results", [])
        self.tasks[new_task_id].results = results
        await broadcast(events.browser_result(new_task_id, results))

        # Re-check again -- browsing can take seconds; another interrupt
        # may have landed while we were building the response.
        if not self.is_current(new_task_id):
            logger.info(f"Task '{new_task_id}' went stale after results arrived, before being spoken. Discarding.")
            return None

        # --- Speak the result (Person 1) ---
        await self._emit_state(new_task_id, TaskState.SPEAKING, broadcast)
        summary = self._build_summary(results, result)
        await broadcast(events.speech_started(new_task_id, summary))
        asyncio.create_task(asyncio.to_thread(rime_service.speak, summary, new_task_id))

        await self._emit_state(new_task_id, TaskState.COMPLETED, broadcast)
        await broadcast(events.task_completed(new_task_id))

        return result

    async def _interrupt(self, old_task_id: str, new_task_id: str, broadcast: BroadcastFn) -> None:
        """
        Tears down the old task across BOTH modules before the new one is
        registered, so there is never a moment where two tasks are both
        considered current.
        """
        if old_task_id in self.tasks:
            self.tasks[old_task_id].state = TaskState.INTERRUPTED

        voice_manager.interrupt_current_task(new_task_id=new_task_id, reason="User interrupted")
        browser_cancel_task(old_task_id)

        await broadcast(events.task_interrupted(old_task_id, new_task_id))
        await broadcast(events.state_changed(old_task_id, TaskState.INTERRUPTED.value))

        # Brief yield so the old task's best-effort Playwright page.close()
        # actually runs before the new task opens its own browser context
        # (avoids a rare Windows/Playwright driver timing race).
        await asyncio.sleep(0.05)

        logger.info(f"Interrupted '{old_task_id}' -> '{new_task_id}'")

    @staticmethod
    def _build_summary(results: list, result: dict) -> str:
        if not results:
            return "I couldn't find anything matching those constraints."
        top = results[0]
        return f"I found {len(results)} matching options. The top pick is {top.get('name')} for {top.get('price')}."

    # --------------------------------------------------------------- misc
    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_current_task(self) -> Optional[Task]:
        return self.tasks.get(self.current_task_id) if self.current_task_id else None

    def reset(self) -> None:
        if self.current_task_id:
            browser_cancel_task(self.current_task_id)
        voice_manager.reset()
        self.current_task_id = None
        self.tasks = {}
        self._last_intent = None


# Single shared instance for the process (one-user hackathon MVP, no DB).
task_manager = TaskManager()
