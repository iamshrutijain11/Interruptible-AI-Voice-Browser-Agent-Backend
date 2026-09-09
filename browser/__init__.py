"""
browser package public interface.

Person 3 (backend/voice integration) should only ever need:

    from browser import search_products, create_task, cancel_task, is_task_valid, get_task_result

Typical flow for one user instruction (including an interrupt):

    task_id = new_unique_id()
    create_task(task_id, query)          # invalidates any previous task
    result = await search_products(query, task_id)   # or run as a background task

To handle an interrupt while a search is in flight:

    cancel_task(old_task_id)             # best-effort stop + guarantees staleness
    new_task_id = new_unique_id()
    create_task(new_task_id, new_query)  # old_task_id is now guaranteed invalid
    result = await search_products(new_query, new_task_id)

No Playwright knowledge required beyond this file.
"""

from .search import search_products
from .task_control import (
    registry,
    create_task,
    is_task_valid,
    cancel_task,
    get_task_result,
    get_task_status,
)

__all__ = [
    "search_products",
    "registry",
    "create_task",
    "is_task_valid",
    "cancel_task",
    "get_task_result",
    "get_task_status",
]
