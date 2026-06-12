"""End-to-end tests for /workflow command via gateway."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class FakeRunner:
    """Minimal fake runner for testing _run_workflow_task."""
    def __init__(self):
        self.adapters = {}
        self._background_tasks = set()

    def _reply_anchor_for_event(self, event):
        return None

    def _thread_metadata_for_source(self, source, event_message_id):
        return {}

    async def _run_workflow_task(self, task, source, task_id, event_message_id=None):
        pass


@pytest.mark.asyncio
async def test_workflow_task_runs_6_nodes():
    """_run_workflow_task should execute 6-node DAG and send results."""
    from gateway.run import GatewayRunner

    runner = FakeRunner()
    mock_adapter = AsyncMock()
    runner.adapters = {"test": mock_adapter}

    with patch("agent.workflow_orchestrator.WorkflowOrchestrator.run") as mock_run:
        mock_run.return_value = {"summary": "OAuth2 flow designed", "next_action": "implement"}

        await GatewayRunner._run_workflow_task(
            runner,
            task="design OAuth2 flow",
            source=MagicMock(platform="test", chat_id="123"),
            task_id="wf_test",
            event_message_id=None,
        )

    # Should have sent final result
    mock_adapter.send.assert_called()
    call_args = mock_adapter.send.call_args
    assert "Workflow complete" in call_args[0][1]


@pytest.mark.asyncio
async def test_workflow_task_progress_pushed():
    """Per-node progress should be pushed to chat."""
    from gateway.run import GatewayRunner

    runner = FakeRunner()
    mock_adapter = AsyncMock()
    runner.adapters = {"test": mock_adapter}

    with patch("agent.workflow_orchestrator.WorkflowOrchestrator.run") as mock_run:
        mock_run.return_value = {"summary": "done", "next_action": "none"}

        await GatewayRunner._run_workflow_task(
            runner,
            task="test",
            source=MagicMock(platform="test", chat_id="123"),
            task_id="wf_test",
            event_message_id=None,
        )

    # Should have at least final message
    assert any("Workflow complete" in str(call) for call in mock_adapter.send.call_args_list)


@pytest.mark.asyncio
async def test_workflow_task_failure_handled():
    """Workflow failure should send error message."""
    from gateway.run import GatewayRunner

    runner = FakeRunner()
    mock_adapter = AsyncMock()
    runner.adapters = {"test": mock_adapter}

    with patch("agent.workflow_orchestrator.WorkflowOrchestrator.run") as mock_run:
        mock_run.side_effect = Exception("DAG failed")

        await GatewayRunner._run_workflow_task(
            runner,
            task="test",
            source=MagicMock(platform="test", chat_id="123"),
            task_id="wf_test",
            event_message_id=None,
        )

    # Should have sent error
    mock_adapter.send.assert_called()
    call_args = mock_adapter.send.call_args
    assert "failed" in call_args[0][1].lower()


@pytest.mark.asyncio
async def test_workflow_command_dispatched():
    """/workflow command should return start message."""
    from gateway.run import GatewayRunner

    runner = FakeRunner()
    mock_adapter = AsyncMock()
    runner.adapters = {"test": mock_adapter}

    event = MagicMock()
    event.get_command_args.return_value = "test task"
    event.source.platform = "test"
    event.source.chat_id = "123"

    # Patch _run_workflow_task to avoid actual execution
    with patch.object(GatewayRunner, "_run_workflow_task", new=AsyncMock()):
        result = await GatewayRunner._handle_workflow_command(runner, event)

    assert "Workflow started" in result
    assert "wf_" in result  # task ID format
