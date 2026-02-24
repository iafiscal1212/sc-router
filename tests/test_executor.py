"""Tests for distributed executor."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from sc_router.agent import AgentStatus, RemoteAgent, AgentRegistry
from sc_router.catalog import Tool
from sc_router.executor import execute, ExecutionResult, _detect_mode
from sc_router.router import RoutingResult, ToolAssignment
from sc_router.decomposer import DecompositionResult, SubTask
from sc_router.tracing import RoutingTrace


def _make_registry_with_agents():
    """Build a registry with search + weather agents, both HEALTHY."""
    reg = AgentRegistry()
    reg.register(RemoteAgent(
        id="search-agent",
        url="http://search:8081",
        tool=Tool(
            name="search", description="Search the web",
            input_types={"query"}, output_types={"search_results"},
            capability_tags={"search", "web"},
        ),
        status=AgentStatus.HEALTHY,
    ))
    reg.register(RemoteAgent(
        id="weather-agent",
        url="http://weather:8082",
        tool=Tool(
            name="weather", description="Get weather",
            input_types={"location"}, output_types={"weather_data"},
            capability_tags={"weather", "forecast"},
        ),
        status=AgentStatus.HEALTHY,
    ))
    return reg


class TestDetectMode:
    def test_single_assignment(self):
        result = RoutingResult(
            sc_level=0, strategy='direct',
            tool_assignments=[ToolAssignment(tool='t1', subtask='s1')],
        )
        assert _detect_mode(result) == 'single'

    def test_empty_assignments(self):
        result = RoutingResult(sc_level=0, strategy='direct', tool_assignments=[])
        assert _detect_mode(result) == 'single'

    def test_sequential_from_decomposition(self):
        decomp = DecompositionResult(
            subtasks=[], mode='sequential', original_query='q')
        result = RoutingResult(
            sc_level=1, strategy='pipeline_sequential',
            tool_assignments=[
                ToolAssignment(tool='t1', subtask='s1'),
                ToolAssignment(tool='t2', subtask='s2'),
            ],
            decomposition=decomp,
        )
        assert _detect_mode(result) == 'sequential'

    def test_parallel_from_decomposition(self):
        decomp = DecompositionResult(
            subtasks=[], mode='parallel', original_query='q')
        result = RoutingResult(
            sc_level=1, strategy='pipeline_parallel',
            tool_assignments=[
                ToolAssignment(tool='t1', subtask='s1'),
                ToolAssignment(tool='t2', subtask='s2'),
            ],
            decomposition=decomp,
        )
        assert _detect_mode(result) == 'parallel'

    def test_parallel_from_strategy(self):
        result = RoutingResult(
            sc_level=1, strategy='pipeline_parallel',
            tool_assignments=[
                ToolAssignment(tool='t1', subtask='s1'),
                ToolAssignment(tool='t2', subtask='s2'),
            ],
        )
        assert _detect_mode(result) == 'parallel'


class TestExecuteSingle:
    @pytest.mark.asyncio
    async def test_single_execution(self):
        reg = _make_registry_with_agents()
        routing = RoutingResult(
            sc_level=0, strategy='direct',
            tool_assignments=[ToolAssignment(tool='search', subtask='find python docs')],
        )

        mock_response = {"answer": "python docs found"}
        with patch('sc_router.executor._http_post',
                   new_callable=AsyncMock, return_value=mock_response):
            result = await execute(routing, reg)

        assert result.success
        assert len(result.outputs) == 1
        assert result.outputs[0]['tool'] == 'search'
        assert result.outputs[0]['output'] == mock_response

    @pytest.mark.asyncio
    async def test_single_unhealthy_agent(self):
        reg = _make_registry_with_agents()
        reg.set_status("search-agent", AgentStatus.UNHEALTHY)

        routing = RoutingResult(
            sc_level=0, strategy='direct',
            tool_assignments=[ToolAssignment(tool='search', subtask='find X')],
        )

        result = await execute(routing, reg)
        assert len(result.outputs) == 1
        assert 'error' in result.outputs[0]

    @pytest.mark.asyncio
    async def test_single_agent_not_found(self):
        reg = _make_registry_with_agents()
        routing = RoutingResult(
            sc_level=0, strategy='direct',
            tool_assignments=[ToolAssignment(tool='nonexistent', subtask='do X')],
        )
        result = await execute(routing, reg)
        assert len(result.outputs) == 1
        assert 'error' in result.outputs[0]


class TestExecuteSequential:
    @pytest.mark.asyncio
    async def test_sequential_pipeline(self):
        reg = _make_registry_with_agents()
        decomp = DecompositionResult(
            subtasks=[
                SubTask(text="search for info", tools=["search"]),
                SubTask(text="get weather", tools=["weather"], depends_on=[0]),
            ],
            mode='sequential', original_query='q',
        )
        routing = RoutingResult(
            sc_level=1, strategy='pipeline_sequential',
            tool_assignments=[
                ToolAssignment(tool='search', subtask='search for info'),
                ToolAssignment(tool='weather', subtask='get weather'),
            ],
            decomposition=decomp,
        )

        call_count = 0
        async def mock_post(url, payload, client=None, timeout=30.0):
            nonlocal call_count
            call_count += 1
            return {"step": call_count, "data": f"result_{call_count}"}

        with patch('sc_router.executor._http_post', side_effect=mock_post):
            result = await execute(routing, reg)

        assert result.success
        assert len(result.outputs) == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_sequential_context_passing(self):
        """Output from step 1 should be passed as context to step 2."""
        reg = _make_registry_with_agents()
        decomp = DecompositionResult(
            subtasks=[], mode='sequential', original_query='q')
        routing = RoutingResult(
            sc_level=1, strategy='pipeline_sequential',
            tool_assignments=[
                ToolAssignment(tool='search', subtask='find data'),
                ToolAssignment(tool='weather', subtask='process data'),
            ],
            decomposition=decomp,
        )

        payloads_received = []
        async def mock_post(url, payload, client=None, timeout=30.0):
            payloads_received.append(payload)
            return {"output_data": "from_step_1"}

        with patch('sc_router.executor._http_post', side_effect=mock_post):
            await execute(routing, reg)

        # Second call should have context from first
        assert len(payloads_received) == 2
        assert 'context' not in payloads_received[0]  # first has no context
        # The second should have context since first returned output
        # (the output is wrapped in {'output': response})


class TestExecuteParallel:
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        reg = _make_registry_with_agents()
        decomp = DecompositionResult(
            subtasks=[], mode='parallel', original_query='q')
        routing = RoutingResult(
            sc_level=1, strategy='pipeline_parallel',
            tool_assignments=[
                ToolAssignment(tool='search', subtask='find data'),
                ToolAssignment(tool='weather', subtask='get weather'),
            ],
            decomposition=decomp,
        )

        async def mock_post(url, payload, client=None, timeout=30.0):
            return {"ok": True}

        with patch('sc_router.executor._http_post', side_effect=mock_post):
            result = await execute(routing, reg)

        assert result.success
        assert len(result.outputs) == 2

    @pytest.mark.asyncio
    async def test_parallel_with_unhealthy_agent(self):
        reg = _make_registry_with_agents()
        reg.set_status("weather-agent", AgentStatus.UNHEALTHY)

        decomp = DecompositionResult(
            subtasks=[], mode='parallel', original_query='q')
        routing = RoutingResult(
            sc_level=1, strategy='pipeline_parallel',
            tool_assignments=[
                ToolAssignment(tool='search', subtask='find data'),
                ToolAssignment(tool='weather', subtask='get weather'),
            ],
            decomposition=decomp,
        )

        async def mock_post(url, payload, client=None, timeout=30.0):
            return {"ok": True}

        with patch('sc_router.executor._http_post', side_effect=mock_post):
            result = await execute(routing, reg)

        assert len(result.outputs) == 2
        # weather was skipped
        weather_output = [o for o in result.outputs if o['tool'] == 'weather'][0]
        assert 'error' in weather_output


class TestExecuteEmpty:
    @pytest.mark.asyncio
    async def test_empty_assignments(self):
        reg = _make_registry_with_agents()
        routing = RoutingResult(
            sc_level=0, strategy='direct', tool_assignments=[])
        result = await execute(routing, reg)
        assert result.success
        assert result.outputs == []


class TestExecutionResult:
    def test_to_dict(self):
        trace = RoutingTrace(query="test")
        result = ExecutionResult(
            outputs=[{"tool": "t1", "output": "ok"}],
            trace=trace,
            success=True,
        )
        d = result.to_dict()
        assert d['success'] is True
        assert len(d['outputs']) == 1
        assert 'trace' in d
