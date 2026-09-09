"""
events.py
---------
WebSocket event protocol. Every event is a plain dict with a "type" field.
React listens for these "type" values and updates UI accordingly.

New events for the upgraded agent:
  plan.created        — task plan with all step names, emitted at task start
  step.changed        — individual step status update as pipeline runs
  recommendation.ready — structured explainable recommendation object
  approval.required   — signals frontend to show Approve/Reject UI
  approval.done       — result of user's approval decision
"""

from typing import Optional, List, Dict, Any


# ── Existing events (unchanged) ─────────────────────────────────────────────

def task_started(task_id: str, query: str, constraints: dict) -> dict:
    return {"type": "task.started", "task_id": task_id,
            "query": query, "constraints": constraints}


def task_interrupted(old_task_id: str, new_task_id: str) -> dict:
    return {"type": "task.interrupted",
            "old_task_id": old_task_id, "new_task_id": new_task_id}


def browser_started(task_id: str) -> dict:
    return {"type": "browser.started", "task_id": task_id}


def browser_result(
    task_id: str,
    results: List[Dict[str, Any]],
    is_fallback: bool = False,
    warning: Optional[str] = None,
    source: Optional[str] = None,
) -> dict:
    return {
        "type": "browser.result",
        "task_id": task_id,
        "results": results,
        "is_fallback": is_fallback,
        "warning": warning,
        "source": source,
    }


def speech_started(task_id: str, text: str, language: str = "en") -> dict:
    return {"type": "speech.started", "task_id": task_id, "text": text, "language": language}


def task_completed(task_id: str) -> dict:
    return {"type": "task.completed", "task_id": task_id}


def task_error(task_id: str, error: str) -> dict:
    return {"type": "task.error", "task_id": task_id, "error": error}


def state_changed(task_id: Optional[str], state: str) -> dict:
    return {"type": "state.changed", "task_id": task_id, "state": state}


def transcript_updated(task_id: Optional[str], text: str,
                       language: str = "en") -> dict:
    """
    Broadcast whenever a user utterance is transcribed.
    `language` is the ISO 639-1 code detected by STT (e.g. "hi", "es").
    """
    return {"type": "transcript.updated", "task_id": task_id,
            "text": text, "language": language}


# ── New events for the upgraded agent ───────────────────────────────────────

def plan_created(task_id: str, steps: List[Dict[str, Any]],
                 is_replan: bool = False) -> dict:
    """
    Emitted once at task start with the full ordered step list.
    `steps` is a list of {id, label, status} dicts.
    `is_replan` is True when this plan was triggered by an interrupt — the
    frontend uses this to show "Re-planning (context preserved)" context.
    """
    return {"type": "plan.created", "task_id": task_id,
            "steps": steps, "is_replan": is_replan}


def step_changed(task_id: str, step_id: str, status: str,
                 detail: Optional[str] = None) -> dict:
    """
    Emitted as each plan step actually starts, completes, fails, or is
    interrupted. Maps 1:1 to real code phase transitions in task_manager.py.

    status: "pending" | "running" | "completed" | "failed" |
            "interrupted" | "skipped"
    """
    return {"type": "step.changed", "task_id": task_id,
            "step_id": step_id, "status": status, "detail": detail}


def recommendation_ready(task_id: str, recommendation: Dict[str, Any]) -> dict:
    """
    Emitted after analysis completes. `recommendation` is a Recommendation
    model serialised to dict — contains top_pick, reason, score_breakdown,
    comparison_summary, and sources_searched.
    """
    return {"type": "recommendation.ready", "task_id": task_id,
            "recommendation": recommendation}


def approval_required(task_id: str, text: str) -> dict:
    """
    Emitted after the agent speaks its recommendation, signalling the frontend
    to show Approve / Reject UI. `text` is the spoken recommendation text so
    the frontend can display it alongside the buttons.
    """
    return {"type": "approval.required", "task_id": task_id, "text": text}


def approval_done(task_id: str, approved: bool, action: Optional[str] = None) -> dict:
    """
    Emitted after the user's approval decision is processed.
    `action` describes what the agent did (e.g. "opened product URL").
    """
    return {"type": "approval.done", "task_id": task_id,
            "approved": approved, "action": action}


def metrics_updated(metrics: dict) -> dict:
    """
    Emitted whenever any session-level metric changes.
    Contains the full snapshot so the frontend can overwrite state without
    tracking deltas.

    Metrics keys:
      current_task      — human-readable search query for the active task
      current_action    — single-sentence description of what the agent is doing right now
      activity          — machine-readable activity label (one of the ACTIVITY constants)
      websites_visited  — how many distinct sites were queried this session
      pages_analyzed    — total products/pages inspected across all tasks
      actions_performed — total pipeline steps completed this session
      records_extracted — total product records extracted this session
      interruptions     — number of user interruptions this session
      replans           — number of re-plans triggered by interruptions
    """
    return {"type": "metrics.updated", "metrics": metrics}

