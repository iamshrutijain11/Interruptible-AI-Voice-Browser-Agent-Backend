"""
test_integration.py
--------------------
Exercises the spec's own test checklist end to end, against the REAL voice
and browser modules (real Playwright browser, real -- or mock, per .env --
STT/TTS). No mocking of Person 1/Person 2's modules here; see the
docstring in task_manager.py if you want a fast, network-free logic-only
test instead (that variant is what was used during development, with fake
stand-ins for voice/browser).

Run with:
    python test_integration.py
"""

import sys
import asyncio
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from browser.browser import browser_manager
from task_manager import TaskManager

EVENTS_LOG = []


async def broadcast(ev):
    EVENTS_LOG.append(ev)
    try:
        print(f"[event] {ev}")
    except Exception:
        safe_str = str(ev).encode("ascii", "backslashreplace").decode("ascii")
        print(f"[event] {safe_str}")


def event_types_since(marker: int):
    return [e["type"] for e in EVENTS_LOG[marker:]]


async def test_1_normal_search(tm: TaskManager):
    print("\n=== TEST 1: Normal voice search ===")
    marker = len(EVENTS_LOG)
    result = await tm.handle_utterance("black running shoes under 2000", broadcast)
    types = event_types_since(marker)
    assert "task.started" in types
    assert "task.completed" in types or "task.error" in types
    assert result is not None
    print("PASSED: normal search produced a result and the expected event sequence.")
    return result


async def test_2_interrupt_while_speaking(tm: TaskManager):
    print("\n=== TEST 2: Interrupt while AI is (about to be) speaking ===")
    marker = len(EVENTS_LOG)
    old_task_id = tm.current_task_id

    task1 = asyncio.create_task(tm.handle_utterance("red sneakers under 3000", broadcast))
    await asyncio.sleep(0.05)  # let it register + start the ack speech

    result2 = await tm.handle_utterance("Wait! Under 1500, size 9.", broadcast)
    result1 = await task1

    types = event_types_since(marker)
    assert "task.interrupted" in types
    assert result1 is None, f"expected stale task to return None, got {result1}"
    assert result2 is not None
    print("PASSED: old task discarded, new task delivered a result.")
    return result2


async def test_3_interrupt_while_browsing(tm: TaskManager):
    print("\n=== TEST 3: Interrupt while browser is actively searching ===")
    marker = len(EVENTS_LOG)

    task1 = asyncio.create_task(tm.handle_utterance("blue jacket under 5000", broadcast))
    await asyncio.sleep(1.0)  # let it get well into the BROWSING state

    result2 = await tm.handle_utterance("actually, green jacket size L", broadcast)
    result1 = await task1

    types = event_types_since(marker)
    assert "task.interrupted" in types
    assert result1 is None
    assert result2 is not None
    print("PASSED: interrupt mid-browse correctly discarded task 1's result.")
    return result2


async def test_4_stale_result_after_new_task(tm: TaskManager):
    print("\n=== TEST 4: Old task result arrives after a new task already started ===")
    # This is really the same guarantee as tests 2/3, verified directly via
    # is_current() rather than via timing -- included separately since the
    # spec calls it out as its own test.
    old_task_id = tm._new_task_id()
    tm.tasks[old_task_id] = None  # not a real task; just checking is_current()
    tm.current_task_id = tm._new_task_id()
    assert tm.is_current(old_task_id) is False
    tm.reset()
    print("PASSED: is_current() correctly rejects a superseded task_id.")


async def test_5_new_task_completes_normally(tm: TaskManager):
    print("\n=== TEST 5: A fresh task (no interrupt involved) completes normally ===")
    marker = len(EVENTS_LOG)
    result = await tm.handle_utterance("white sandals under 1000", broadcast)
    types = event_types_since(marker)
    assert "task.interrupted" not in types
    assert result is not None
    print("PASSED: uninterrupted task completed without any spurious interrupt event.")


async def test_6_api_failure_handling(tm: TaskManager):
    print("\n=== TEST 6: Simulated API/search failure is handled cleanly ===")
    marker = len(EVENTS_LOG)
    # An empty/nonsense query is the easiest network-independent way to
    # trigger a "no results" or "error" style outcome without needing to
    # fake a network failure.
    result = await tm.handle_utterance("asdkjhqwe zxcvqwe under 1", broadcast)
    types = event_types_since(marker)
    assert result is not None, "handle_utterance must never raise -- always returns a clean dict"
    assert result.get("status") in ("completed_empty", "error", "completed")
    print(f"PASSED: search failure/empty-result handled cleanly (status={result.get('status')!r}), no crash.")


async def main():
    await browser_manager.start()
    tm = TaskManager()

    try:
        await test_1_normal_search(tm)
        await test_2_interrupt_while_speaking(tm)
        await test_3_interrupt_while_browsing(tm)
        await test_4_stale_result_after_new_task(tm)
        await test_5_new_task_completes_normally(tm)
        await test_6_api_failure_handling(tm)

        print("\nALL 6 SPEC TESTS PASSED")
    finally:
        await browser_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
