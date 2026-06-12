"""OpenViking context engine plugin.

Stores compacted middle conversation turns in OpenViking and replaces them
with a small retrieval handoff, avoiding LLM summarization for context
management while keeping the OpenViking tools available for expansion.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List

from agent.context_engine import ContextEngine
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH, estimate_messages_tokens_rough
from plugins.memory.openviking import _DEFAULT_ENDPOINT, _VikingClient

logger = logging.getLogger(__name__)

HANDOFF_PREFIX = "[OPENVIKING CONTEXT HANDOFF — REFERENCE ONLY]"


class OpenVikingContextEngine(ContextEngine):
    """Context engine backed by OpenViking semantic storage/retrieval."""

    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 8

    def __init__(self) -> None:
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.context_length = int(os.environ.get("OPENVIKING_CONTEXT_LENGTH", "128000") or 128000)
        self.threshold_tokens = max(int(self.context_length * self.threshold_percent), MINIMUM_CONTEXT_LENGTH)
        self.compression_count = 0
        self._endpoint = os.environ.get("OPENVIKING_ENDPOINT", _DEFAULT_ENDPOINT)
        self._api_key = os.environ.get("OPENVIKING_API_KEY", "")
        self._account = os.environ.get("OPENVIKING_ACCOUNT", "default")
        self._user = os.environ.get("OPENVIKING_USER", "default")
        self._agent = os.environ.get("OPENVIKING_AGENT", "hermes")
        self._session_id = ""
        self._client: _VikingClient | None = None
        self._last_error = ""
        self._stored_uris: list[str] = []
        self._connect()

    @property
    def name(self) -> str:
        return "openviking"

    def is_available(self) -> bool:
        return bool(self._client)

    def _connect(self) -> None:
        try:
            self._client = _VikingClient(
                self._endpoint,
                self._api_key,
                account=self._account,
                user=self._user,
                agent=self._agent,
            )
            if not self._client.health():
                self._last_error = f"OpenViking server at {self._endpoint} is not reachable"
                logger.warning(self._last_error)
                self._client = None
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("OpenViking context engine disabled: %s", exc)
            self._client = None

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or self._session_id or uuid.uuid4().hex
        if kwargs.get("context_length"):
            self.update_model("", int(kwargs["context_length"]))
        if self._client is None:
            self._connect()

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._stored_uris = []
        self._last_error = ""

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self.context_length = context_length or self.context_length
        self.threshold_tokens = max(int(self.context_length * self.threshold_percent), MINIMUM_CONTEXT_LENGTH)

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_total_tokens = int(usage.get("total_tokens", self.last_prompt_tokens + self.last_completion_tokens) or 0)

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return bool(tokens and tokens >= self.threshold_tokens and self._client is not None)

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        if not self._client:
            return False
        try:
            return estimate_messages_tokens_rough(messages) >= self.threshold_tokens
        except Exception:
            return False

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        start, end = self._middle_bounds(messages)
        return end > start

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        if not messages or not self._client:
            return messages
        start, end = self._middle_bounds(messages)
        if end <= start:
            return messages

        assert self._client is not None
        middle = messages[start:end]
        compact_uri = self._write_middle_turns(middle, focus_topic=focus_topic)
        related = self._search_related(focus_topic or self._query_from_tail(messages[end:]))
        handoff = self._handoff_message(compact_uri, len(middle), related)
        self.compression_count += 1
        self._stored_uris.append(compact_uri)
        return messages[:start] + [handoff] + messages[end:]

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "engine": self.name,
            "endpoint": self._endpoint,
            "stored_uris": len(self._stored_uris),
            "available": bool(self._client),
        })
        if self._last_error:
            status["last_error"] = self._last_error
        return status

    def _middle_bounds(self, messages: List[Dict[str, Any]]) -> tuple[int, int]:
        if not messages:
            return 0, 0
        start = 1 if messages[0].get("role") == "system" else 0
        non_system_seen = 0
        while start < len(messages) and non_system_seen < self.protect_first_n:
            if messages[start].get("role") != "system":
                non_system_seen += 1
            start += 1
        end = max(start, len(messages) - self.protect_last_n)
        start = self._align_forward(messages, start)
        end = self._align_backward(messages, end)
        return start, end

    @staticmethod
    def _align_forward(messages: List[Dict[str, Any]], idx: int) -> int:
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    @staticmethod
    def _align_backward(messages: List[Dict[str, Any]], idx: int) -> int:
        while idx > 0 and idx < len(messages) and messages[idx].get("role") == "tool":
            idx -= 1
        return idx

    def _write_middle_turns(self, middle: List[Dict[str, Any]], *, focus_topic: str | None = None) -> str:
        uri = self._build_context_uri()
        payload = {
            "session_id": self._session_id,
            "compressed_at": int(time.time()),
            "focus_topic": focus_topic or "",
            "messages": middle,
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        self._client.post("/api/v1/content/write", {
            "uri": uri,
            "content": content,
            "mode": "create",
        })
        return uri

    def _search_related(self, query: str, limit: int = 5) -> list[dict]:
        if not query:
            return []
        try:
            resp = self._client.post("/api/v1/search/find", {"query": query, "top_k": limit})
            result = resp.get("result", {})
            rows = []
            for bucket in ("memories", "resources", "skills"):
                for item in result.get(bucket, [])[:limit]:
                    rows.append({
                        "uri": item.get("uri", ""),
                        "type": bucket.rstrip("s"),
                        "score": item.get("score", 0),
                        "abstract": item.get("abstract", ""),
                    })
            rows.sort(key=lambda r: r.get("score") or 0, reverse=True)
            return rows[:limit]
        except Exception as exc:
            logger.debug("OpenViking context search failed: %s", exc)
            return []

    @staticmethod
    def _query_from_tail(tail: List[Dict[str, Any]]) -> str:
        parts = []
        for msg in tail[-6:]:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip()[:500])
        return "\n".join(parts)[-2000:]

    def _handoff_message(self, uri: str, dropped_count: int, related: list[dict]) -> Dict[str, Any]:
        lines = [
            HANDOFF_PREFIX,
            "Earlier middle conversation turns were stored in OpenViking, not summarized.",
            f"Stored URI: {uri}",
            f"Stored messages: {dropped_count}",
            "Use viking_read with the URI for full details; use viking_search for semantic recall.",
        ]
        if related:
            lines.append("Relevant OpenViking context:")
            for item in related[:5]:
                score = item.get("score") or 0
                abstract = (item.get("abstract") or "").replace("\n", " ")[:300]
                lines.append(f"- [{score:.2f}] {abstract} ({item.get('uri','')})")
        return {"role": "system", "content": "\n".join(lines)}

    def _build_context_uri(self) -> str:
        sid = (self._session_id or "session").replace("/", "_")[:80]
        slug = uuid.uuid4().hex[:12]
        return f"viking://user/{self._user}/agent/{self._agent}/context/{sid}/compact_{slug}.json"


def register(ctx) -> None:
    ctx.register_context_engine(OpenVikingContextEngine())
