"""
task_manager.py
----------------
Central orchestrator with a full 8-step visible task plan.
Includes session-level metrics tracked across tasks and broadcast as
metrics.updated events so the frontend Activity Panel updates in real time.

Metrics are SESSION-scoped (survive across tasks, reset only on session reset):
  interruptions, replans, websites_visited, pages_analyzed,
  actions_performed, records_extracted

Metrics are TASK-scoped (reset on each new task):
  current_task, current_action, activity

Every call to handle_utterance() creates a fresh PlanStep list broadcast
over WebSocket as plan.created. As each pipeline phase starts and finishes,
step.changed events fire — these map 1:1 to real code execution, never faked.

8 Plan Steps (in order):
  1. understand  — agent.parse_utterance()
  2. plan        — routing / site selection
  3. browse      — search_products() browser automation
  4. extract     — results arrive from browser
  5. analyze     — agent.build_recommendation()
  6. recommend   — recommendation_ready + TTS spoken summary
  7. approve     — approval_required emitted, waiting for user
  8. act         — open product URL or confirm action

On interrupt: the running step gets status="interrupted", the new task's
plan starts fresh with is_replan=True. The interrupt/staleness logic
(is_current()) is purely task_id-based and completely language/plan-agnostic.

Approval flow:
  - After step 6 (recommend), the manager emits approval_required and
    sets self._awaiting_approval = task_id.
  - handle_approval(approved, broadcast) is called by main.py's WebSocket
    handler when the user sends {action:"approval", approved:bool}.
  - Approval opens the top product URL (if any) in a new browser tab.
  - Rejection re-runs parse_utterance with "rejected" context and re-plans.
"""

import uuid
import asyncio
import logging
from typing import Optional, Dict, Callable, Awaitable, Any, List

import events
import agent
from models import (
    Task, TaskState, ParsedIntent,
    PlanStep, StepStatus, PLAN_TEMPLATE,
)

from voice import voice_manager, rime_service
from browser import (
    search_products,
    create_task as browser_create_task,
    cancel_task as browser_cancel_task,
)

logger = logging.getLogger("task_manager")

BroadcastFn = Callable[[dict], Awaitable[None]]


# Activity labels matching the UI panel states (safe/high-level only)
ACTIVITY = {
    "idle":             "Idle",
    "listening":        "Listening",
    "understanding":    "Understanding",
    "planning":         "Planning",
    "browsing":         "Browsing",
    "extracting":       "Extracting data",
    "processing":       "Processing",
    "reasoning":        "Reasoning",
    "waiting_approval": "Waiting for approval",
    "executing":        "Executing",
    "completed":        "Completed",
    "interrupted":      "Interrupted — re-planning",
    "error":            "Error",
}


class TaskManager:
    def __init__(self):
        self.current_task_id: Optional[str] = None
        self.tasks: Dict[str, Task] = {}
        self._last_intent: Optional[ParsedIntent] = None
        self._awaiting_approval: Optional[str] = None
        self._approval_event: Optional[asyncio.Event] = None
        self._approval_result: Optional[bool] = None
        # ── Session-level metrics (persist across tasks, reset on reset()) ──
        self._m = {
            "current_task":      "",
            "current_action":    "Ready",
            "activity":          "idle",
            "websites_visited":  0,
            "pages_analyzed":    0,
            "actions_performed": 0,
            "records_extracted": 0,
            "interruptions":     0,
            "replans":           0,
        }

    # ──────────────────────────────────────────── core versioning
    def is_current(self, task_id: str) -> bool:
        return task_id is not None and task_id == self.current_task_id

    def _new_task_id(self) -> str:
        return f"task_{uuid.uuid4().hex[:8]}"

    # ──────────────────────────────────────────── metrics helpers
    async def _metric(self, broadcast: BroadcastFn, **kwargs):
        """Update one or more metrics fields and broadcast the full snapshot."""
        self._m.update(kwargs)
        await broadcast(events.metrics_updated(dict(self._m)))

    # ──────────────────────────────────────────── plan helpers
    def _make_plan(self) -> List[PlanStep]:
        return [PlanStep(id=s["id"], label=s["label"]) for s in PLAN_TEMPLATE]

    async def _step_start(self, task_id: str, step_id: str,
                          broadcast: BroadcastFn, detail: str = None):
        if not self.is_current(task_id):
            return
        task = self.tasks.get(task_id)
        if task:
            for s in task.plan:
                if s.id == step_id:
                    s.status = StepStatus.RUNNING
                    s.detail = detail
        await broadcast(events.step_changed(task_id, step_id, "running", detail))

    async def _step_done(self, task_id: str, step_id: str,
                         broadcast: BroadcastFn, detail: str = None):
        if not self.is_current(task_id):
            return
        task = self.tasks.get(task_id)
        if task:
            for s in task.plan:
                if s.id == step_id:
                    s.status = StepStatus.COMPLETED
                    s.detail = detail
        await broadcast(events.step_changed(task_id, step_id, "completed", detail))

    async def _step_fail(self, task_id: str, step_id: str,
                         broadcast: BroadcastFn, detail: str = None):
        task = self.tasks.get(task_id)
        if task:
            for s in task.plan:
                if s.id == step_id:
                    s.status = StepStatus.FAILED
                    s.detail = detail
        await broadcast(events.step_changed(task_id, step_id, "failed", detail))

    async def _step_interrupt(self, task_id: str, step_id: str, broadcast: BroadcastFn):
        task = self.tasks.get(task_id)
        if task:
            for s in task.plan:
                if s.status == StepStatus.RUNNING:
                    s.status = StepStatus.INTERRUPTED
        await broadcast(events.step_changed(task_id, step_id, "interrupted"))

    async def _emit_state(self, task_id: str, state: TaskState, broadcast: BroadcastFn):
        if not self.is_current(task_id):
            return
        self.tasks[task_id].state = state
        await broadcast(events.state_changed(task_id, state.value))

    # ──────────────────────────────────────────── main pipeline
    async def handle_utterance(
        self,
        text: str,
        broadcast: BroadcastFn,
        stt_language: str = "en",
    ) -> Optional[dict]:
        """
        Full 8-step pipeline. Each step fires step.changed before/after execution.
        Returns the final result dict, or None if the task went stale.
        """
        old_task_id = self.current_task_id
        running_step = None   # track which step is running for interrupt marking

        # Transcript broadcast (with language for badge)
        await broadcast(events.transcript_updated(old_task_id, text, language=stt_language))
        await self._metric(broadcast, activity="understanding",
                           current_action="Listening to your instruction")

        # ── Cancel any pending approval from a previous task ──────────────────
        if self._awaiting_approval and self._approval_event:
            self._approval_result = None
            self._approval_event.set()
        self._awaiting_approval = None
        self._approval_event = None

        # ── Step 1: UNDERSTAND ───────────────────────────────────────────────
        new_task_id = self._new_task_id()
        plan = self._make_plan()
        is_replan = old_task_id is not None
        if is_replan:
            self._m["replans"] += 1

        # Interrupt old task before registering new one
        if old_task_id:
            await self._interrupt(old_task_id, new_task_id, broadcast, running_step)

        self.tasks[new_task_id] = Task(
            task_id=new_task_id,
            query="...",  # updated after understand step
            constraints=self._last_intent.constraints if self._last_intent else
                        __import__("models").Constraints(),
            state=TaskState.THINKING,
            plan=plan,
        )
        self.current_task_id = new_task_id

        # Emit plan + task.started
        await broadcast(events.plan_created(
            new_task_id,
            [{"id": s.id, "label": s.label, "status": s.status.value} for s in plan],
            is_replan=is_replan,
        ))
        await broadcast(events.state_changed(new_task_id, TaskState.THINKING.value))

        running_step = "understand"
        await self._metric(broadcast, activity="understanding",
                           current_action="Parsing intent with language model")
        await self._step_start(new_task_id, "understand", broadcast,
                               "Parsing your intent with LLM")

        intent = await agent.parse_utterance(text, self._last_intent,
                                             stt_language=stt_language)
        self._last_intent = intent
        detected_lang = intent.detected_language or "en"
        search_query  = agent.build_search_query(intent)

        if not self.is_current(new_task_id):
            return None

        self.tasks[new_task_id].query = search_query
        self.tasks[new_task_id].constraints = intent.constraints
        self._m["actions_performed"] += 1
        await self._step_done(new_task_id, "understand", broadcast,
                              f"'{intent.query}' ({detected_lang})")

        # task.started after we know the real query
        await broadcast(events.task_started(
            new_task_id, search_query, intent.constraints.dict()
        ))

        # ── Step 2: PLAN (routing) ───────────────────────────────────────────
        running_step = "plan"
        await self._emit_state(new_task_id, TaskState.PLANNING, broadcast)
        await self._metric(broadcast, activity="planning",
                           current_action="Selecting which websites to search",
                           current_task=search_query)
        await self._step_start(new_task_id, "plan", broadcast, "Selecting search sites")

        from browser.routing import get_parallel_sites_for_query
        sites = get_parallel_sites_for_query(search_query)
        self._m["actions_performed"] += 1
        await self._step_done(new_task_id, "plan", broadcast,
                              f"Searching: {', '.join(sites)}")

        # ── Acknowledgement TTS (while browsing starts) ─────────────────────
        await self._emit_state(new_task_id, TaskState.SPEAKING, broadcast)
        ack_en = f"Searching for {search_query}..."
        ack    = await agent.translate_text(ack_en, detected_lang)
        await broadcast(events.speech_started(new_task_id, ack))
        asyncio.create_task(
            asyncio.to_thread(rime_service.speak, ack, new_task_id, lang=detected_lang)
        )

        # ── Step 3: BROWSE ───────────────────────────────────────────────────
        running_step = "browse"
        await self._emit_state(new_task_id, TaskState.BROWSING, broadcast)
        await self._metric(broadcast, activity="browsing",
                           current_action=f"Automating browser on {', '.join(sites)}")
        await self._step_start(new_task_id, "browse", broadcast,
                               f"Automating {len(sites)} site(s)")
        await broadcast(events.browser_started(new_task_id))
        browser_create_task(new_task_id, search_query)
        self._m["websites_visited"] += len(sites)

        result = await search_products(search_query, new_task_id,
                                       constraints=intent.constraints)

        if not self.is_current(new_task_id) or result.get("status") == "stale":
            return None

        if result.get("status") not in ("completed", "completed_empty"):
            error_msg = result.get("error", "Unknown search error")
            self.tasks[new_task_id].error = error_msg
            await self._step_fail(new_task_id, "browse", broadcast, error_msg)
            await self._emit_state(new_task_id, TaskState.ERROR, broadcast)
            await broadcast(events.task_error(new_task_id, error_msg))
            err_text = await agent.translate_text(
                "Sorry, I ran into a problem while searching.", detected_lang)
            asyncio.create_task(
                asyncio.to_thread(rime_service.speak, err_text, new_task_id,
                                  lang=detected_lang)
            )
            return result

        self._m["actions_performed"] += 1
        await self._step_done(new_task_id, "browse", broadcast,
                              f"{len(result.get('results', []))} raw results")

        # ── Step 4: EXTRACT ──────────────────────────────────────────────────
        running_step = "extract"
        await self._metric(broadcast, activity="extracting",
                           current_action="Extracting product records from pages")
        await self._step_start(new_task_id, "extract", broadcast)

        results = result.get("results", [])
        sources = result.get("sources", result.get("source", "live"))
        if isinstance(sources, str):
            sources = [sources]

        self.tasks[new_task_id].results = results
        await broadcast(events.browser_result(new_task_id, results))
        self._m["records_extracted"]  += len(results)
        self._m["pages_analyzed"]     += len(results)
        self._m["actions_performed"]  += 1
        await self._step_done(new_task_id, "extract", broadcast,
                              f"{len(results)} products extracted")

        if not self.is_current(new_task_id):
            return None

        # ── Step 5: ANALYZE ──────────────────────────────────────────────────
        running_step = "analyze"
        await self._emit_state(new_task_id, TaskState.ANALYZING, broadcast)
        await self._metric(broadcast, activity="processing",
                           current_action=f"Scoring & comparing {len(results)} products")
        await self._step_start(new_task_id, "analyze", broadcast,
                               "Scoring & comparing products")

        recommendation = await agent.build_recommendation(
            results, search_query, intent.constraints, sources
        )
        self.tasks[new_task_id].recommendation = recommendation.dict()
        self._m["actions_performed"] += 1
        await self._step_done(new_task_id, "analyze", broadcast,
                              f"Top pick: {recommendation.top_pick_name[:30]}")
        await self._metric(broadcast, activity="reasoning",
                           current_action=f"Building explanation for {recommendation.top_pick_name[:30]}")

        await broadcast(events.recommendation_ready(
            new_task_id, recommendation.dict()
        ))

        if not self.is_current(new_task_id):
            return None

        # ── Step 6: RECOMMEND (speak) ────────────────────────────────────────
        running_step = "recommend"
        await self._emit_state(new_task_id, TaskState.RECOMMENDING, broadcast)
        await self._step_start(new_task_id, "recommend", broadcast)

        summary_en = self._build_summary(results, result, recommendation)
        summary    = await agent.translate_text(summary_en, detected_lang)

        await broadcast(events.speech_started(new_task_id, summary))
        asyncio.create_task(
            asyncio.to_thread(rime_service.speak, summary, new_task_id,
                              lang=detected_lang)
        )
        self._m["actions_performed"] += 1
        await self._step_done(new_task_id, "recommend", broadcast)

        if not self.is_current(new_task_id):
            return None

        # ── Step 7: APPROVE ──────────────────────────────────────────────────
        running_step = "approve"
        if not results or not recommendation or not recommendation.top_pick_name:
            await self._step_start(new_task_id, "approve", broadcast, "No products to approve")
            await self._step_done(new_task_id, "approve", broadcast, "Completed — 0 matches")
            await self._step_start(new_task_id, "act", broadcast, "No action required")
            await self._step_done(new_task_id, "act", broadcast, "No action required")
            await self._metric(broadcast, activity="completed", current_action="Search finished — no products found")
            await self._emit_state(new_task_id, TaskState.COMPLETED, broadcast)
            await broadcast(events.task_completed(new_task_id))
            return result

        await self._emit_state(new_task_id, TaskState.AWAITING_APPROVAL, broadcast)
        await self._metric(broadcast, activity="waiting_approval",
                           current_action="Waiting for your approval")
        await self._step_start(new_task_id, "approve", broadcast,
                               "Say 'yes' to open the product, or 'no' to refine")

        approval_text_en = (
            f"I recommend {recommendation.top_pick_name}. "
            "Say yes to open it, or no to refine your search."
        )
        approval_text = await agent.translate_text(approval_text_en, detected_lang)
        await broadcast(events.approval_required(new_task_id, approval_text))

        # Set up approval event — wait up to 15 seconds for user response
        self._awaiting_approval = new_task_id
        ev = asyncio.Event()
        self._approval_event  = ev
        self._approval_result = None

        try:
            await asyncio.wait_for(ev.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.info(f"[task_manager] Approval timed out for {new_task_id}; auto-completing")

        if not self.is_current(new_task_id):
            return None

        approved = self._approval_result
        self._awaiting_approval = None
        self._approval_event    = None
        self._approval_result   = None

        await self._step_done(new_task_id, "approve", broadcast,
                              "Approved" if approved else "Declined / timed out")

        # ── Step 8: ACT ──────────────────────────────────────────────────────
        running_step = "act"
        action_taken = None

        if approved and results and recommendation.top_pick_index < len(results):
            await self._metric(broadcast, activity="executing",
                               current_action="Opening product in browser")
            await self._step_start(new_task_id, "act", broadcast,
                                   "Opening product in browser")
            top_url = results[recommendation.top_pick_index].get("url")
            if top_url:
                try:
                    from browser.browser import browser_manager
                    ctx = await browser_manager.new_context()
                    pg  = await ctx.new_page()
                    await pg.goto(top_url, wait_until="commit", timeout=15000)
                    action_taken = f"Opened: {top_url[:60]}"
                    logger.info(f"[task_manager] Opened product URL: {top_url}")
                except Exception as e:
                    logger.warning(f"[task_manager] Could not open URL: {e}")
                    action_taken = "Could not open URL"
            else:
                action_taken = "No URL available"

            act_en = f"Done! I've opened {recommendation.top_pick_name} for you."
            act_text = await agent.translate_text(act_en, detected_lang)
            asyncio.create_task(
                asyncio.to_thread(rime_service.speak, act_text, new_task_id,
                                  lang=detected_lang)
            )
        else:
            await self._step_start(new_task_id, "act", broadcast)
            action_taken = "Skipped (no approval)"

        self._m["actions_performed"] += 1
        await self._step_done(new_task_id, "act", broadcast, action_taken)
        await broadcast(events.approval_done(new_task_id, bool(approved), action_taken))

        # ── Complete ─────────────────────────────────────────────────────────
        await self._metric(broadcast, activity="completed",
                           current_action="Task completed successfully")
        await self._emit_state(new_task_id, TaskState.COMPLETED, broadcast)
        await broadcast(events.task_completed(new_task_id))
        return result

    # ──────────────────────────────────────────── approval handler
    async def handle_approval(self, approved: bool, broadcast: BroadcastFn):
        """Called by main.py's WebSocket handler when user sends approval decision."""
        if self._awaiting_approval and self._approval_event:
            self._approval_result = approved
            self._approval_event.set()
            logger.info(f"[task_manager] Approval received: approved={approved} "
                        f"for task {self._awaiting_approval}")
        else:
            logger.warning("[task_manager] Approval received but no task is awaiting it")

    # ──────────────────────────────────────────── interrupt
    async def _interrupt(self, old_task_id: str, new_task_id: str,
                         broadcast: BroadcastFn, running_step: str = None):
        if old_task_id in self.tasks:
            self.tasks[old_task_id].state = TaskState.INTERRUPTED
            if running_step:
                await self._step_interrupt(old_task_id, running_step, broadcast)

        voice_manager.interrupt_current_task(new_task_id=new_task_id,
                                             reason="User interrupted")
        browser_cancel_task(old_task_id)

        self._m["interruptions"] += 1
        await self._metric(broadcast, activity="interrupted",
                           current_action="Interrupt received — re-planning")
        await broadcast(events.task_interrupted(old_task_id, new_task_id))
        await broadcast(events.state_changed(old_task_id, TaskState.INTERRUPTED.value))
        await asyncio.sleep(0.05)
        logger.info(f"Interrupted '{old_task_id}' -> '{new_task_id}'")

    # ──────────────────────────────────────────── summary builder
    @staticmethod
    def _build_summary(results: list, result: dict, recommendation=None) -> str:
        """English summary for TTS. Always English — translate_text() handles localisation."""
        if not results:
            return "I couldn't find anything matching those constraints."
        top = results[recommendation.top_pick_index] if recommendation else results[0]
        name  = top.get("name", "the top product")
        price = top.get("price", "")
        site  = top.get("site", "")
        n     = len(results)
        s = f"I found {n} option{'s' if n != 1 else ''}. "
        s += f"My top recommendation is {name}"
        if price:
            s += f" at {price}"
        if site:
            s += f" on {site}"
        s += "."
        if recommendation and recommendation.reason:
            s += f" {recommendation.reason}"
        return s

    # ──────────────────────────────────────────── helpers
    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_current_task(self) -> Optional[Task]:
        return self.tasks.get(self.current_task_id) if self.current_task_id else None

    def reset(self) -> None:
        if self.current_task_id:
            browser_cancel_task(self.current_task_id)
        if self._awaiting_approval and self._approval_event:
            self._approval_result = None
            self._approval_event.set()
        voice_manager.reset()
        self.current_task_id    = None
        self.tasks              = {}
        self._last_intent       = None
        self._awaiting_approval = None
        self._approval_event    = None
        self._approval_result   = None
        # Reset all session metrics
        self._m = {
            "current_task":      "",
            "current_action":    "Ready",
            "activity":          "idle",
            "websites_visited":  0,
            "pages_analyzed":    0,
            "actions_performed": 0,
            "records_extracted": 0,
            "interruptions":     0,
            "replans":           0,
        }


task_manager = TaskManager()
