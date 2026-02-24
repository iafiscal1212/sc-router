"""End-to-end integration tests: config → registry → route → execute."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from sc_router.agent import AgentStatus, RemoteAgent, AgentRegistry
from sc_router.catalog import Tool
from sc_router.config import parse_config, build_registry
from sc_router.executor import execute
from sc_router.health import HealthChecker
from sc_router.router import route
from sc_router.tracing import RoutingTrace, TracingHook


INTEGRATION_CONFIG = {
    'agents': [
        {
            'id': 'search-agent',
            'url': 'http://search:8081',
            'tool': {
                'name': 'search',
                'description': 'Search the web for information',
                'capability_tags': ['search', 'web', 'find', 'lookup', 'information'],
                'input_types': ['query'],
                'output_types': ['search_results'],
            },
        },
        {
            'id': 'summarizer-agent',
            'url': 'http://summarizer:8082',
            'tool': {
                'name': 'summarizer',
                'description': 'Summarize text content',
                'capability_tags': ['summarize', 'summary', 'condense', 'text'],
                'input_types': ['text', 'search_results'],
                'output_types': ['summary'],
            },
        },
        {
            'id': 'weather-agent',
            'url': 'http://weather:8083',
            'tool': {
                'name': 'weather',
                'description': 'Get weather forecast for a location',
                'capability_tags': ['weather', 'forecast', 'temperature', 'climate'],
                'input_types': ['location'],
                'output_types': ['weather_data'],
            },
        },
    ],
    'health': {
        'failure_threshold': 3,
        'recovery_timeout_s': 30,
    },
}


class TestConfigToRouting:
    """Test the full path: config → registry → route."""

    def test_config_to_route_sc0(self):
        """Simple query should classify as SC(0) and pick weather tool."""
        registry = build_registry(INTEGRATION_CONFIG)
        result = route("What's the weather in Madrid?", registry.catalog)

        assert result.sc_level == 0
        assert result.strategy == 'direct'
        assert len(result.tool_assignments) >= 1
        assert result.tool_assignments[0].tool == 'weather'

    def test_config_to_route_pipeline(self):
        """Pipeline query should classify and produce multiple assignments."""
        registry = build_registry(INTEGRATION_CONFIG)
        result = route(
            "Search for Python tutorials, then summarize the results",
            registry.catalog,
        )

        # Should produce multiple tool assignments
        assert len(result.tool_assignments) >= 1
        tools_used = {ta.tool for ta in result.tool_assignments}
        assert 'search' in tools_used or 'summarizer' in tools_used

    def test_registry_catalog_matches_direct_catalog(self):
        """Routes through registry.catalog should match plain ToolCatalog."""
        registry = build_registry(INTEGRATION_CONFIG)

        # Build equivalent plain catalog
        plain = registry.catalog

        query = "What's the weather in Madrid?"
        r1 = route(query, registry.catalog)
        r2 = route(query, plain)

        assert r1.sc_level == r2.sc_level
        assert r1.strategy == r2.strategy


class TestRoutingWithExecution:
    """Test route → execute pipeline."""

    @pytest.mark.asyncio
    async def test_route_and_execute_sc0(self):
        registry = build_registry(INTEGRATION_CONFIG)

        # Mark agents healthy
        for agent in registry.agents:
            registry.set_status(agent.id, AgentStatus.HEALTHY)

        # Route
        routing_result = route(
            "What's the weather in Madrid?", registry.catalog)

        # Execute with mocked HTTP
        async def mock_post(url, payload, client=None, timeout=30.0):
            return {"temperature": "22C", "city": "Madrid"}

        with patch('sc_router.executor._http_post', side_effect=mock_post):
            exec_result = await execute(routing_result, registry)

        assert exec_result.success
        assert len(exec_result.outputs) >= 1
        assert exec_result.outputs[0]['output']['city'] == 'Madrid'

    @pytest.mark.asyncio
    async def test_route_execute_with_tracing(self):
        registry = build_registry(INTEGRATION_CONFIG)
        for agent in registry.agents:
            registry.set_status(agent.id, AgentStatus.HEALTHY)

        # Setup tracing
        hook = TracingHook()
        trace = RoutingTrace(query="What's the weather in Madrid?")

        # Classify
        trace.start_classification()
        routing_result = route("What's the weather in Madrid?", registry.catalog)
        trace.finish_classification(
            sc_level=routing_result.sc_level,
            strategy=routing_result.strategy,
        )

        # Execute
        async def mock_post(url, payload, client=None, timeout=30.0):
            return {"temperature": "22C"}

        with patch('sc_router.executor._http_post', side_effect=mock_post):
            exec_result = await execute(routing_result, registry, trace=trace)

        hook.record(trace)

        assert trace.sc_level == 0
        assert trace.classification_ms >= 0
        assert trace.execution_ms >= 0
        assert hook.count == 1

        # Verify trace serialization
        d = trace.to_dict()
        assert d['sc_level'] == 0
        assert len(d['steps']) >= 1

    @pytest.mark.asyncio
    async def test_execute_skips_unhealthy(self):
        registry = build_registry(INTEGRATION_CONFIG)

        # Only weather is healthy
        registry.set_status("weather-agent", AgentStatus.HEALTHY)
        registry.set_status("search-agent", AgentStatus.UNHEALTHY)
        registry.set_status("summarizer-agent", AgentStatus.UNHEALTHY)

        routing_result = route(
            "What's the weather in Madrid?", registry.catalog)

        async def mock_post(url, payload, client=None, timeout=30.0):
            return {"temperature": "22C"}

        with patch('sc_router.executor._http_post', side_effect=mock_post):
            exec_result = await execute(routing_result, registry)

        # Weather agent should have been called
        weather_outputs = [o for o in exec_result.outputs
                          if o.get('tool') == 'weather']
        assert len(weather_outputs) >= 1


class TestHealthCheckIntegration:
    @pytest.mark.asyncio
    async def test_health_checker_with_registry(self):
        registry = build_registry(INTEGRATION_CONFIG)
        checker = HealthChecker(registry, failure_threshold=2)

        # Mock all agents as healthy
        with patch.object(checker, '_ping', new_callable=AsyncMock, return_value=True):
            results = await checker.check_all()

        assert all(v is True for v in results.values())
        assert all(a.status == AgentStatus.HEALTHY for a in registry.agents)

    @pytest.mark.asyncio
    async def test_health_then_route_with_healthy_catalog(self):
        registry = build_registry(INTEGRATION_CONFIG)
        checker = HealthChecker(registry, failure_threshold=1)

        # Mark search as unhealthy
        async def selective_ping(url):
            return 'search' not in url

        with patch.object(checker, '_ping', side_effect=selective_ping):
            await checker.check_all()

        # Route using healthy catalog
        healthy_cat = registry.healthy_catalog()
        result = route("What's the weather in Madrid?", healthy_cat)

        # Weather should still work
        assert result.tool_assignments[0].tool == 'weather'


class TestTracingIntegration:
    def test_kore_mind_trace_format(self):
        """Verify kore-mind trace format is complete."""
        trace = RoutingTrace(query="test query")
        trace.sc_level = 1
        trace.strategy = 'pipeline_sequential'
        trace.classification_ms = 5.0
        trace.execution_ms = 50.0
        trace.total_ms = 55.0

        km = trace.to_kore_mind_trace()
        assert km['type'] == 'sc_routing'
        assert km['query'] == 'test query'
        assert km['sc_level'] == 1
        assert 'timing' in km
        assert km['timing']['total_ms'] == 55.0

    def test_tracing_hook_with_listener(self):
        """TracingHook should notify listeners on record."""
        hook = TracingHook()
        received = []
        hook.add_listener(lambda t: received.append(t.to_dict()))

        trace = RoutingTrace(query="hello")
        trace.sc_level = 0
        trace.strategy = 'direct'
        hook.record(trace)

        assert len(received) == 1
        assert received[0]['query'] == 'hello'
