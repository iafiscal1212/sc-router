"""Tests for the main routing engine."""

import time

from sc_router import ToolCatalog, Tool, route
from sc_router.router import RoutingResult


class TestRoute:
    def test_sc0_direct_dispatch(self, sample_catalog):
        result = route("What is the weather in Madrid?", sample_catalog)
        assert isinstance(result, RoutingResult)
        assert result.sc_level == 0
        assert result.strategy == 'direct'
        assert len(result.tool_assignments) >= 1
        assert result.tool_assignments[0].tool == 'weather'

    def test_sc0_calculator(self, sample_catalog):
        result = route("Calculate 2 + 2", sample_catalog)
        assert result.sc_level == 0
        assert result.strategy == 'direct'
        assert any(a.tool == 'calculator' for a in result.tool_assignments)

    def test_sc1_pipeline(self, sample_catalog):
        result = route(
            "First search the web for news, then summarize the results",
            sample_catalog,
        )
        assert result.sc_level in (0, 1)
        assert len(result.tool_assignments) >= 1

    def test_sc2_returns_multiple_tools(self, sample_catalog):
        result = route(
            "Plan a trip: find flights to Paris, search hotels near the center, "
            "find restaurants, and optimize within $2000 budget",
            sample_catalog,
        )
        assert result.sc_level >= 1
        assert len(result.tool_assignments) >= 2

    def test_routing_result_fields(self, sample_catalog):
        result = route("What is the weather?", sample_catalog)
        assert hasattr(result, 'sc_level')
        assert hasattr(result, 'strategy')
        assert hasattr(result, 'tool_assignments')
        assert hasattr(result, 'classification')

    def test_agent_callback(self, sample_catalog):
        called = {}

        def my_agent(query, catalog, classification):
            called['invoked'] = True
            return RoutingResult(
                sc_level=3,
                strategy='custom_agent',
                tool_assignments=[],
                classification=classification,
            )

        result = route(
            "Analyze market trends, cross-reference with social media sentiment, "
            "correlate the data, and build a predictive model from combined insights",
            sample_catalog,
            agent_callback=my_agent,
        )
        # If classified as SC3, callback should be used
        if result.sc_level == 3:
            assert called.get('invoked', False)
            assert result.strategy == 'custom_agent'

    def test_latency_under_50ms(self, sample_catalog):
        """All classifications should complete in <50ms."""
        queries = [
            "What is the weather?",
            "Search and summarize news",
            "Plan trip with flights, hotels, restaurants, budget $2000",
            "Analyze trends, cross-reference sentiment, build model",
        ]
        for query in queries:
            start = time.perf_counter()
            route(query, sample_catalog)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 50, f"Query took {elapsed_ms:.1f}ms: {query[:40]}"
