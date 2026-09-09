"""
events.py
---------
The WebSocket event protocol. Every event is a plain dict with a "type"
field, matching the exact shapes from the integration spec. React listens
for these "type" values and updates the UI accordingly -- see
frontend/src/websocket.js for the matching client-side switch.

Nothing in here has side effects; these are pure builder functions. main.py
and task_manager.py call these and broadcast the result over the socket.
"""

from typing import Optional, List, Dict, Any


def task_started(task_id: str, query: str, constraints: dict) -> dict:
    return {"type": "task.started", "task_id": task_id, "query": query, "constraints": constraints}


def task_interrupted(old_task_id: str, new_task_id: str) -> dict:
    return {"type": "task.interrupted", "old_task_id": old_task_id, "new_task_id": new_task_id}


def browser_started(task_id: str) -> dict:
    return {"type": "browser.started", "task_id": task_id}


def browser_result(task_id: str, results: List[Dict[str, Any]]) -> dict:
    return {"type": "browser.result", "task_id": task_id, "results": results}


def speech_started(task_id: str, text: str) -> dict:
    return {"type": "speech.started", "task_id": task_id, "text": text}


def task_completed(task_id: str) -> dict:
    return {"type": "task.completed", "task_id": task_id}


def task_error(task_id: str, error: str) -> dict:
    return {"type": "task.error", "task_id": task_id, "error": error}


def state_changed(task_id: Optional[str], state: str) -> dict:
    return {"type": "state.changed", "task_id": task_id, "state": state}


def transcript_updated(task_id: Optional[str], text: str) -> dict:
    return {"type": "transcript.updated", "task_id": task_id, "text": text}
