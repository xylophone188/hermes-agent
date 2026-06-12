"""Tests for OpenViking context engine plugin."""

from __future__ import annotations

import json

from agent.context_engine import ContextEngine
from plugins.context_engine import load_context_engine
from plugins.context_engine.openviking import HANDOFF_PREFIX, OpenVikingContextEngine


class FakeVikingClient:
    writes = []
    searches = []

    def __init__(self, *args, **kwargs):
        pass

    def health(self):
        return True

    def post(self, path, payload=None, **kwargs):
        payload = payload or {}
        if path == "/api/v1/content/write":
            self.writes.append(payload)
            return {"result": {"uri": payload.get("uri")}}
        if path == "/api/v1/search/find":
            self.searches.append(payload)
            return {
                "result": {
                    "memories": [
                        {
                            "uri": "viking://user/default/agent/hermes/memories/patterns/mem_1.md",
                            "score": 0.91,
                            "abstract": "Relevant prior pattern",
                        }
                    ],
                    "resources": [],
                    "skills": [],
                }
            }
        return {"result": {}}


def _messages(n=16):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"message {i}"})
    return msgs


def test_openviking_context_engine_loads(monkeypatch):
    FakeVikingClient.writes = []
    monkeypatch.setattr("plugins.context_engine.openviking._VikingClient", FakeVikingClient)
    engine = OpenVikingContextEngine()
    assert isinstance(engine, ContextEngine)
    assert engine.name == "openviking"
    assert engine.is_available() is True


def test_loader_discovers_openviking(monkeypatch):
    FakeVikingClient.writes = []
    monkeypatch.setattr("plugins.context_engine.openviking._VikingClient", FakeVikingClient)
    engine = load_context_engine("openviking")
    assert engine is not None
    assert engine.name == "openviking"


def test_compress_stores_middle_and_injects_handoff(monkeypatch):
    FakeVikingClient.writes = []
    FakeVikingClient.searches = []
    monkeypatch.setattr("plugins.context_engine.openviking._VikingClient", FakeVikingClient)
    engine = OpenVikingContextEngine()
    engine.on_session_start("sess-1")
    before = _messages(18)

    after = engine.compress(before, focus_topic="patterns")

    assert len(after) < len(before)
    assert engine.compression_count == 1
    assert len(FakeVikingClient.writes) == 1
    write = FakeVikingClient.writes[0]
    assert "/context/sess-1/" in write["uri"]
    assert write["uri"].startswith("viking://user/")
    stored = json.loads(write["content"])
    assert stored["session_id"] == "sess-1"
    assert stored["focus_topic"] == "patterns"
    assert stored["messages"]
    handoff = [m for m in after if m.get("content", "").startswith(HANDOFF_PREFIX)]
    assert len(handoff) == 1
    assert write["uri"] in handoff[0]["content"]
    assert "Relevant prior pattern" in handoff[0]["content"]
    assert FakeVikingClient.searches[0]["query"] == "patterns"
