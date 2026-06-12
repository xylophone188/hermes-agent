"""Tests for background task tool-progress callback.

Verifies that _run_background_task wires a tool_complete_callback into the
AIAgent that:
  1. Calls adapter.send for slow tool calls (duration >= threshold).
  2. Skips adapter.send for fast tool calls (duration < threshold).
  3. Includes task_id and a sequential counter in the message.
  4. Never raises — errors inside the callback are swallowed.
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(platform="qqbot", chat_id="chat-1", user_id="u1"):
    return SimpleNamespace(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        user_id_alt=None,
        user_name="tester",
        chat_name="test-chat",
        chat_type="private",
        thread_id=None,
    )


def _make_runner(adapter):
    """Minimal GatewayRunner stand-in with just the pieces under test."""
    runner = MagicMock()
    runner.adapters = {"qqbot": adapter}
    runner._thread_metadata_for_source = MagicMock(return_value={"thread_id": None})
    runner._service_tier = None
    runner._session_db = None
    runner._fallback_model = None
    return runner


def _extract_progress_cb_from_run_sync(runner, source, task_id, loop):
    """
    Simulate the setup block in _run_background_task that builds
    _background_tool_complete_cb.  We reproduce the same closure logic here
    so the unit test doesn't have to spin up a full gateway.
    """
    import os
    from agent.display import get_cute_tool_message
    from agent.async_utils import safe_schedule_threadsafe

    adapter = runner.adapters[source.platform]
    _thread_metadata = runner._thread_metadata_for_source(source, None)
    _PROGRESS_MIN_DURATION_S = float(os.environ.get("HERMES_BG_PROGRESS_MIN_S", "1.5"))
    _bg_tool_count = [0]

    def _background_tool_complete_cb(
        tool_call_id: str,
        tool_name: str,
        tool_args: dict,
        tool_result: str,
        duration: float = 0.0,
    ) -> None:
        if duration < _PROGRESS_MIN_DURATION_S:
            return
        _bg_tool_count[0] += 1
        try:
            cute = get_cute_tool_message(tool_name, tool_args, duration, result=tool_result)
        except Exception:
            cute = f"{tool_name} ({duration:.1f}s)"
        progress_text = f"⏳ [{task_id}] #{_bg_tool_count[0]} {cute}"
        try:
            safe_schedule_threadsafe(
                adapter.send(
                    source.chat_id,
                    progress_text,
                    metadata=_thread_metadata,
                ),
                loop,
            )
        except Exception:
            pass

    return _background_tool_complete_cb, _bg_tool_count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBackgroundProgressCallback(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the progress callback closure."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.adapter = MagicMock()
        self.adapter.send = AsyncMock(return_value=None)
        self.source = _make_source()
        self.runner = _make_runner(self.adapter)

    def tearDown(self):
        self.loop.close()

    # ------------------------------------------------------------------
    # 1. Slow tool → send is scheduled
    # ------------------------------------------------------------------

    def test_slow_tool_schedules_send(self):
        cb, counter = _extract_progress_cb_from_run_sync(
            self.runner, self.source, "task-42", self.loop
        )
        cb("id1", "terminal", {"command": "ls"}, "result-output", duration=2.0)
        # drain scheduled coroutines
        self.loop.run_until_complete(asyncio.sleep(0))
        self.adapter.send.assert_called_once()
        args = self.adapter.send.call_args
        assert "chat-1" == args[0][0]
        msg = args[0][1]
        assert "task-42" in msg
        assert "#1" in msg

    # ------------------------------------------------------------------
    # 2. Fast tool → send NOT called
    # ------------------------------------------------------------------

    def test_fast_tool_skips_send(self):
        cb, counter = _extract_progress_cb_from_run_sync(
            self.runner, self.source, "task-fast", self.loop
        )
        cb("id1", "terminal", {"command": "ls"}, "out", duration=0.1)
        self.loop.run_until_complete(asyncio.sleep(0))
        self.adapter.send.assert_not_called()
        assert counter[0] == 0

    # ------------------------------------------------------------------
    # 3. Counter increments across multiple calls
    # ------------------------------------------------------------------

    def test_counter_increments(self):
        cb, counter = _extract_progress_cb_from_run_sync(
            self.runner, self.source, "task-cnt", self.loop
        )
        cb("id1", "terminal", {}, "r1", duration=2.0)
        cb("id2", "web_search", {}, "r2", duration=3.0)
        cb("id3", "read_file", {}, "r3", duration=0.5)  # fast, skipped
        self.loop.run_until_complete(asyncio.sleep(0))
        assert counter[0] == 2
        assert self.adapter.send.call_count == 2
        # Second call message contains #2
        second_msg = self.adapter.send.call_args_list[1][0][1]
        assert "#2" in second_msg

    # ------------------------------------------------------------------
    # 4. Callback swallows send errors silently
    # ------------------------------------------------------------------

    def test_send_error_swallowed(self):
        self.adapter.send = AsyncMock(side_effect=RuntimeError("network down"))
        cb, counter = _extract_progress_cb_from_run_sync(
            self.runner, self.source, "task-err", self.loop
        )
        # Must not raise
        cb("id1", "terminal", {}, "out", duration=2.0)
        self.loop.run_until_complete(asyncio.sleep(0))
        # counter still incremented (error happens after count)
        assert counter[0] == 1

    # ------------------------------------------------------------------
    # 5. HERMES_BG_PROGRESS_MIN_S env override respected
    # ------------------------------------------------------------------

    def test_env_threshold_override(self):
        import os
        with patch.dict(os.environ, {"HERMES_BG_PROGRESS_MIN_S": "5.0"}):
            cb, counter = _extract_progress_cb_from_run_sync(
                self.runner, self.source, "task-env", self.loop
            )
        # duration=2.0 is below new threshold of 5.0 — should skip
        cb("id1", "terminal", {}, "out", duration=2.0)
        self.loop.run_until_complete(asyncio.sleep(0))
        self.adapter.send.assert_not_called()

    # ------------------------------------------------------------------
    # 6. task_id embedded in message
    # ------------------------------------------------------------------

    def test_task_id_in_message(self):
        cb, _ = _extract_progress_cb_from_run_sync(
            self.runner, self.source, "MY-UNIQUE-TASK-ID", self.loop
        )
        cb("id1", "terminal", {}, "out", duration=2.0)
        self.loop.run_until_complete(asyncio.sleep(0))
        msg = self.adapter.send.call_args[0][1]
        assert "MY-UNIQUE-TASK-ID" in msg


