"""
task_control.py
----------------
Single source of truth for task lifecycle / versioning.

Concepts
--------
- Only ONE task is ever "current" at a time (this matches the voice-interrupt
  use case: a new instruction supersedes whatever the browser is doing).
- Creating a new task automatically invalidates the previous current task.
- Cancelling a task marks it cancelled AND (if possible) closes the Playwright
  page belonging to it, to stop real browser work early.
- Validity is checked at multiple points inside search.py so that stale work
  is abandoned as soon as possible, and stale RESULTS are never returned,
  even if the browser finishes navigating/extracting after being superseded.

This module has NO Playwright imports at module scope (only a local import
inside cancel_task for closing a page), so Person 3 can import and reason
about it without knowing anything about browser automation.
"""

import time
import threading
from typing import Optional, Dict, Any


class TaskState:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    INVALID = "invalid"
    ERROR = "error"


class _TaskRecord:
    def __init__(self, task_id: str, query: str):
        self.task_id = task_id
        self.query = query
        self.status = TaskState.PENDING
        self.created_at = time.time()
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        # Playwright Page tied to this task, kept as "Any" so this module
        # never needs to import playwright types.
        self.page = None


class TaskRegistry:
    """
    Guarded with a simple lock. FastAPI + asyncio run on one event loop
    thread here, so this isn't defending against real multi-threading --
    it just keeps read/modify/write sequences atomic and easy to reason
    about.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: Dict[str, _TaskRecord] = {}
        self._current_task_id: Optional[str] = None

    # ---------------------------------------------------------------- create
    def create_task(self, task_id: str, query: str) -> None:
        """Register a new task and invalidate whatever was current before it.

        This is THE mechanism behind "Task 001 becomes INVALID when Task 002
        is created" -- call this every time a new user instruction arrives
        (including interrupts), before calling search_products().
        """
        with self._lock:
            if self._current_task_id and self._current_task_id in self._tasks:
                prev = self._tasks[self._current_task_id]
                if prev.status in (TaskState.PENDING, TaskState.RUNNING):
                    prev.status = TaskState.INVALID

            self._tasks[task_id] = _TaskRecord(task_id, query)
            self._current_task_id = task_id

    # --------------------------------------------------------------- update
    def mark_running(self, task_id: str) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec and rec.status == TaskState.PENDING:
                rec.status = TaskState.RUNNING

    def attach_page(self, task_id: str, page) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec:
                rec.page = page

    def mark_completed(self, task_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec and rec.status not in (TaskState.CANCELLED, TaskState.INVALID):
                rec.status = TaskState.COMPLETED
                rec.result = result

    def mark_error(self, task_id: str, error: str) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec and rec.status not in (TaskState.CANCELLED, TaskState.INVALID):
                rec.status = TaskState.ERROR
                rec.error = error

    # ------------------------------------------------------------- cancel
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Marks a task cancelled and, if a live Playwright page is attached,
        attempts to close it so in-flight navigation aborts immediately.

        Never raises -- safe to call freely from an API endpoint the moment
        a user interrupts.

        NOTE on the distinction the spec calls out:
          - Setting status=CANCELLED (and is_task_valid() -> False) is what
            GUARANTEES stale results are discarded. This always happens.
          - Actually closing the Playwright page is a best-effort attempt at
            REAL execution cancellation. It usually works, but even if it
            didn't, the guarantee above still holds.
        """
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return {"task_id": task_id, "cancelled": False, "reason": "unknown_task"}

            page = rec.page
            rec.status = TaskState.CANCELLED

        physically_stopped = False
        if page is not None:
            try:
                if not page.is_closed():
                    import asyncio
                    # Fire-and-forget close. If a coroutine is currently
                    # awaiting navigation/extraction on this page, Playwright
                    # will raise inside that await (search.py treats this as
                    # a cancellation, not a crash).
                    asyncio.create_task(page.close())
                    physically_stopped = True
            except Exception:
                physically_stopped = False

        return {
            "task_id": task_id,
            "cancelled": True,
            "execution_physically_stopped": physically_stopped,
        }

    # --------------------------------------------------------------- read
    def is_task_valid(self, task_id: str) -> bool:
        """
        A task is valid only if it is still the CURRENT task and has not
        been explicitly cancelled or superseded/invalidated.
        """
        with self._lock:
            if task_id != self._current_task_id:
                return False
            rec = self._tasks.get(task_id)
            if not rec:
                return False
            return rec.status not in (TaskState.CANCELLED, TaskState.INVALID)

    def get_status(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return {"task_id": task_id, "status": "unknown"}
            return {
                "task_id": task_id,
                "status": rec.status,
                "is_current": task_id == self._current_task_id,
            }

    def get_result(self, task_id: str) -> Dict[str, Any]:
        """
        Safe accessor for Person 3 / the API layer. Returns the stored
        result ONLY if the task is still valid (current + not cancelled/
        invalid). Stale results from superseded tasks are never returned
        here, even if they physically finished executing in the background.
        """
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return {"task_id": task_id, "status": "unknown", "results": []}

            if task_id != self._current_task_id:
                return {"task_id": task_id, "status": "stale", "results": []}

            if rec.status == TaskState.CANCELLED:
                return {"task_id": task_id, "status": "cancelled", "results": []}

            if rec.status == TaskState.INVALID:
                return {"task_id": task_id, "status": "invalid", "results": []}

            if rec.status == TaskState.ERROR:
                return {"task_id": task_id, "status": "error", "error": rec.error, "results": []}

            if rec.status == TaskState.COMPLETED and rec.result is not None:
                return rec.result

            return {"task_id": task_id, "status": rec.status, "results": []}


# A single shared registry instance for the whole process.
registry = TaskRegistry()


# ------------------------------------------------------------------------
# Convenience module-level functions -- this is the interface Person 3
# should actually import. No Playwright knowledge required.
# ------------------------------------------------------------------------

def create_task(task_id: str, query: str) -> None:
    registry.create_task(task_id, query)


def is_task_valid(task_id: str) -> bool:
    return registry.is_task_valid(task_id)


def cancel_task(task_id: str) -> Dict[str, Any]:
    return registry.cancel_task(task_id)


def get_task_result(task_id: str) -> Dict[str, Any]:
    return registry.get_result(task_id)


def get_task_status(task_id: str) -> Dict[str, Any]:
    return registry.get_status(task_id)
