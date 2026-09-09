"""
browser.py
----------
Owns the single Playwright browser instance for the process.

Design choices for hackathon reliability:
- ONE browser launched once, reused across every task (fast).
- ONE fresh BrowserContext per task (cheap isolation: no shared cookies/
  cache between searches, and a context/page can be closed independently
  to cancel one task without tearing down the whole browser).
- headless=False by default so judges can SEE the browser open, type,
  and extract results live. Set HEADLESS=true to run headless (e.g. CI
  or a headless server box).

Person 3 never needs to touch this file directly.
"""

import os
import sys
import asyncio
from typing import Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Playwright


class BrowserManager:
    def __init__(self):
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _is_headless() -> bool:
        headless_env = os.getenv("HEADLESS")
        if headless_env is not None:
            return headless_env.lower() == "true"
        # Always headless on Linux / Cloud containers unless explicitly told otherwise
        return True if sys.platform != "win32" else False

    async def start(self) -> None:
        async with self._lock:
            if self._browser is not None:
                return
            headless = self._is_headless()
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=headless,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                    "--start-maximized",
                ],
            )

    async def new_context(self) -> BrowserContext:
        if self._browser is None:
            await self.start()

        headless = self._is_headless()

        context = await self._browser.new_context(
            viewport={"width": 1280, "height": 900} if headless else None,
            no_viewport=not headless,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        context.set_default_navigation_timeout(12000)
        context.set_default_timeout(6000)
        return context

    async def shutdown(self) -> None:
        async with self._lock:
            if self._browser is not None:
                try:
                    await self._browser.close()
                except Exception:
                    pass
                self._browser = None
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None


# Shared singleton used by search.py and main.py.
browser_manager = BrowserManager()
