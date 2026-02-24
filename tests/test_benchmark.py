"""Benchmark: clasificación SC en <50ms con el stack distribuido encima.

Verifica que añadir agent/registry/tracing no degrada el rendimiento
del core de clasificación.
"""

import statistics
import time

import pytest

from sc_router import Tool, ToolCatalog, route
from sc_router.agent import AgentRegistry, AgentStatus, RemoteAgent
from sc_router.tracing import RoutingTrace, TracingHook


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QUERIES = {
    'SC(0) simple': "What's the weather in Madrid?",
    'SC(0) calc': "Calculate 15 * 37",
    'SC(1) pipeline': "Search the web for Python news, then summarize the results",
    'SC(1) translate': "Search for the latest AI papers, then translate to Spanish",
    'SC(2) travel': (
        "Plan a trip: find flights to Paris, search hotels near the center, "
        "find good restaurants, and optimize everything within a $2000 budget"
    ),
    'SC(2) constrained': (
        "Search for budget flights to Rome, find hotels under $100, "
        "look for vegetarian restaurants, and optimize the total cost"
    ),
    'SC(3) entangled': (
        "Analyze market trends, cross-reference with social media sentiment, "
        "correlate the data across multiple sources, and build a predictive "
        "model from the combined insights"
    ),
    'SC(3) complex': (
        "Search for climate data, analyze temperature trends, cross-reference "
        "with economic indicators, summarize findings, translate the report, "
        "and optimize the presentation for stakeholders"
    ),
}


def _build_10_tool_catalog() -> ToolCatalog:
    """El mismo catálogo de 10 tools de conftest."""
    catalog = ToolCatalog()
    tools = [
        ("weather", "Get weather forecast for a location",
         {"location"}, {"weather_data"},
         {"weather", "forecast", "temperature", "climate"}),
        ("calculator", "Perform arithmetic calculations",
         {"expression"}, {"number"},
         {"math", "calculate", "arithmetic", "number"}),
        ("search", "Search the web for information",
         {"query"}, {"search_results"},
         {"search", "web", "find", "lookup", "information"}),
        ("summarizer", "Summarize text content",
         {"text", "search_results"}, {"summary"},
         {"summarize", "summary", "condense", "text"}),
        ("translator", "Translate text between languages",
         {"text", "summary"}, {"translated_text"},
         {"translate", "language", "translation"}),
        ("flight_search", "Search for flights between cities",
         {"origin", "destination", "date"}, {"flight_list"},
         {"flight", "flights", "travel", "book", "airline"}),
        ("hotel_search", "Search for hotels in a city",
         {"city", "date"}, {"hotel_list"},
         {"hotel", "hotels", "accommodation", "travel", "book"}),
        ("restaurant_search", "Find restaurants in a location",
         {"location"}, {"restaurant_list"},
         {"restaurant", "restaurants", "food", "dining"}),
        ("budget_optimizer", "Optimize spending within a budget constraint",
         {"flight_list", "hotel_list", "restaurant_list", "number"},
         {"optimized_plan"},
         {"budget", "optimize", "cost", "plan", "spending"}),
        ("sentiment_analyzer", "Analyze sentiment of text and social media data",
         {"text", "search_results"}, {"sentiment_data"},
         {"sentiment", "analyze", "social", "opinion", "trend"}),
    ]
    for name, desc, inp, out, tags in tools:
        catalog.register(Tool(
            name=name, description=desc,
            input_types=inp, output_types=out, capability_tags=tags,
        ))
    return catalog


def _build_registry_from_catalog(catalog: ToolCatalog) -> AgentRegistry:
    """Envuelve cada tool del catálogo en un RemoteAgent HEALTHY."""
    registry = AgentRegistry()
    for i, tool in enumerate(catalog.tools):
        agent = RemoteAgent(
            id=f"{tool.name}-agent",
            url=f"http://{tool.name}:{8080 + i}",
            tool=tool,
            status=AgentStatus.HEALTHY,
        )
        registry.register(agent)
    return registry


# ---------------------------------------------------------------------------
# Tests: latencia absoluta < 50 ms
# ---------------------------------------------------------------------------

class TestClassificationLatency:
    """Cada query individual debe clasificar en <50ms."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.catalog = _build_10_tool_catalog()

    @pytest.mark.parametrize("label,query", list(QUERIES.items()))
    def test_route_under_50ms(self, label, query):
        start = time.perf_counter()
        result = route(query, self.catalog)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50, (
            f"[{label}] tardó {elapsed_ms:.2f}ms (limit 50ms)")


class TestRegistryClassificationLatency:
    """Mismo test pero usando registry.catalog (path distribuido)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        catalog = _build_10_tool_catalog()
        self.registry = _build_registry_from_catalog(catalog)

    @pytest.mark.parametrize("label,query", list(QUERIES.items()))
    def test_route_via_registry_under_50ms(self, label, query):
        start = time.perf_counter()
        result = route(query, self.registry.catalog)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50, (
            f"[{label}] vía registry tardó {elapsed_ms:.2f}ms (limit 50ms)")


class TestTracingOverhead:
    """Medir el overhead de crear y poblar un RoutingTrace."""

    @pytest.fixture(autouse=True)
    def setup(self):
        catalog = _build_10_tool_catalog()
        self.registry = _build_registry_from_catalog(catalog)

    @pytest.mark.parametrize("label,query", list(QUERIES.items()))
    def test_route_with_tracing_under_50ms(self, label, query):
        trace = RoutingTrace(query=query)
        trace.start_classification()
        result = route(query, self.registry.catalog)
        trace.finish_classification(
            sc_level=result.sc_level,
            strategy=result.strategy,
            confidence=result.classification.get('confidence', ''),
        )
        assert trace.classification_ms < 50, (
            f"[{label}] con tracing tardó {trace.classification_ms:.2f}ms")


class TestHealthyCatalogOverhead:
    """healthy_catalog() filtra en tiempo despreciable."""

    def test_healthy_catalog_build_under_1ms(self):
        catalog = _build_10_tool_catalog()
        registry = _build_registry_from_catalog(catalog)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            hc = registry.healthy_catalog()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg = statistics.mean(times)
        p99 = sorted(times)[98]
        assert avg < 1.0, f"healthy_catalog() avg {avg:.3f}ms (limit 1ms)"
        assert p99 < 2.0, f"healthy_catalog() p99 {p99:.3f}ms (limit 2ms)"


# ---------------------------------------------------------------------------
# Benchmark estadístico: N iteraciones por query
# ---------------------------------------------------------------------------

class TestStatisticalBenchmark:
    """100 iteraciones por query — reporta avg, p50, p95, p99, max."""

    ITERATIONS = 100

    @pytest.fixture(autouse=True)
    def setup(self):
        catalog = _build_10_tool_catalog()
        self.registry = _build_registry_from_catalog(catalog)

    @pytest.mark.parametrize("label,query", list(QUERIES.items()))
    def test_statistical_latency(self, label, query):
        times = []
        for _ in range(self.ITERATIONS):
            start = time.perf_counter()
            route(query, self.registry.catalog)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        times.sort()
        avg = statistics.mean(times)
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[int(len(times) * 0.99)]
        mx = times[-1]

        # Reportar en el nombre del assert para visibilidad
        summary = (f"[{label}] avg={avg:.2f}ms p50={p50:.2f}ms "
                   f"p95={p95:.2f}ms p99={p99:.2f}ms max={mx:.2f}ms")

        assert p99 < 50, f"p99 >= 50ms: {summary}"
        assert avg < 25, f"avg >= 25ms: {summary}"

        # Print para pytest -v -s
        print(f"\n  {summary}")


class TestCatalogScaling:
    """Verificar que el rendimiento se mantiene con catálogos más grandes."""

    @pytest.mark.parametrize("num_tools", [10, 25, 50])
    def test_scaling_under_50ms(self, num_tools):
        catalog = ToolCatalog()
        for i in range(num_tools):
            catalog.register(Tool(
                name=f"tool_{i}",
                description=f"Tool number {i} for testing scalability",
                input_types={f"input_{i}", f"input_{i % 3}"},
                output_types={f"output_{i}", f"output_{(i+1) % num_tools}"},
                capability_tags={f"tag_{i}", f"tag_{i % 5}", "common"},
            ))

        queries = [
            "What's the weather?",
            "Search and then summarize",
            "Find flights, hotels, restaurants, optimize budget",
        ]

        for query in queries:
            times = []
            for _ in range(20):
                start = time.perf_counter()
                route(query, catalog)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            p95 = sorted(times)[int(len(times) * 0.95)]
            assert p95 < 50, (
                f"{num_tools} tools, p95={p95:.2f}ms: {query[:40]}")
