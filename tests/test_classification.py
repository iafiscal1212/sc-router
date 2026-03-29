"""Tests for SC classification with real-world queries.

Covers all 4 SC levels using realistic queries evaluated against
a realistic 10-tool catalog. NO hardcoded features — all queries
run through the full classify_query() pipeline.

Includes regression tests for the SC(3) bug fix (queries with
sequential markers + cross-reference/analysis markers must NOT
be short-circuited to SC(1)).
"""

import pytest

from sc_router import Tool, ToolCatalog, classify_query
from sc_router.features import extract_query_features, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def realistic_catalog():
    """12-tool catalog covering diverse capabilities."""
    catalog = ToolCatalog()
    tools = [
        Tool(
            name="web_search",
            description="Search the web for information",
            input_types={"query"},
            output_types={"search_results"},
            capability_tags={"search", "web", "find", "lookup", "information", "news"},
        ),
        Tool(
            name="calculator",
            description="Perform arithmetic calculations",
            input_types={"expression"},
            output_types={"number"},
            capability_tags={"math", "calculate", "arithmetic", "number"},
        ),
        Tool(
            name="weather",
            description="Get weather forecast for a location",
            input_types={"location"},
            output_types={"weather_data"},
            capability_tags={"weather", "forecast", "temperature", "climate"},
        ),
        Tool(
            name="summarizer",
            description="Summarize text content",
            input_types={"text", "search_results"},
            output_types={"summary"},
            capability_tags={"summarize", "summary", "condense", "text"},
        ),
        Tool(
            name="translator",
            description="Translate text between languages",
            input_types={"text", "summary"},
            output_types={"translated_text"},
            capability_tags={"translate", "language", "translation"},
        ),
        Tool(
            name="flight_search",
            description="Search for flights between cities",
            input_types={"origin", "destination", "date"},
            output_types={"flight_list"},
            capability_tags={"flight", "flights", "travel", "book", "airline"},
        ),
        Tool(
            name="hotel_search",
            description="Search for hotels in a city",
            input_types={"city", "date"},
            output_types={"hotel_list"},
            capability_tags={"hotel", "hotels", "accommodation", "travel", "book"},
        ),
        Tool(
            name="restaurant_search",
            description="Find restaurants in a location",
            input_types={"location"},
            output_types={"restaurant_list"},
            capability_tags={"restaurant", "restaurants", "food", "dining"},
        ),
        Tool(
            name="budget_optimizer",
            description="Optimize spending within a budget constraint",
            input_types={"flight_list", "hotel_list", "restaurant_list", "number"},
            output_types={"optimized_plan"},
            capability_tags={"budget", "optimize", "cost", "plan", "spending"},
        ),
        Tool(
            name="sentiment_analyzer",
            description="Analyze sentiment of text and social media data",
            input_types={"text", "search_results"},
            output_types={"sentiment_data"},
            capability_tags={"sentiment", "analyze", "social", "opinion", "trend"},
        ),
        Tool(
            name="trend_predictor",
            description="Build predictive models from time series data",
            input_types={"sentiment_data", "search_results", "number"},
            output_types={"prediction", "model"},
            capability_tags={"predict", "forecast", "model", "trend", "data"},
        ),
        Tool(
            name="report_generator",
            description="Generate formatted reports from multiple data sources",
            input_types={"summary", "prediction", "optimized_plan", "sentiment_data"},
            output_types={"report"},
            capability_tags={"report", "generate", "format", "document"},
        ),
    ]
    for t in tools:
        catalog.register(t)
    return catalog


# ---------------------------------------------------------------------------
# SC(0): Direct dispatch — simple questions, single tool needed
# ---------------------------------------------------------------------------

class TestSC0DirectDispatch:
    """SC(0) queries should classify at level 0."""

    def test_simple_question(self, realistic_catalog):
        result = classify_query("What is the weather in Madrid?", realistic_catalog)
        assert result['level'] == 0, f"Expected SC(0), got SC({result['level']})"

    def test_arithmetic(self, realistic_catalog):
        result = classify_query("Calculate 15 * 37", realistic_catalog)
        assert result['level'] == 0, f"Expected SC(0), got SC({result['level']})"

    def test_short_imperative(self, realistic_catalog):
        result = classify_query("Search for Python tutorials", realistic_catalog)
        assert result['level'] <= 1, f"Expected SC(0-1), got SC({result['level']})"

    def test_spanish_simple(self, realistic_catalog):
        result = classify_query("Buscar vuelos a Barcelona", realistic_catalog)
        assert result['level'] <= 1, f"Expected SC(0-1), got SC({result['level']})"


# ---------------------------------------------------------------------------
# SC(1): Decomposable pipeline — sequential or parallel, clear steps
# ---------------------------------------------------------------------------

class TestSC1Pipeline:
    """SC(1) queries should classify at level 0 or 1."""

    def test_sequential_search_summarize(self, realistic_catalog):
        result = classify_query(
            "First search for AI news, then summarize the key findings",
            realistic_catalog,
        )
        assert result['level'] in (0, 1), f"Expected SC(0-1), got SC({result['level']})"

    def test_search_then_translate(self, realistic_catalog):
        result = classify_query(
            "Search for the latest AI research papers, then translate the "
            "abstract to Spanish",
            realistic_catalog,
        )
        assert result['level'] in (0, 1), f"Expected SC(0-1), got SC({result['level']})"

    def test_parallel_simple(self, realistic_catalog):
        result = classify_query(
            "Search for flights to Paris and also check hotels near the center",
            realistic_catalog,
        )
        assert result['level'] <= 2, f"Expected SC(0-2), got SC({result['level']})"

    def test_spanish_pipeline(self, realistic_catalog):
        result = classify_query(
            "Primero buscar noticias sobre IA, luego resumir los resultados",
            realistic_catalog,
        )
        assert result['level'] in (0, 1), f"Expected SC(0-1), got SC({result['level']})"


# ---------------------------------------------------------------------------
# SC(2): Constrained search — multiple constraints, ambiguous tool matching
# ---------------------------------------------------------------------------

class TestSC2ConstrainedSearch:
    """SC(2) queries should classify at level >= 1 (at least decomposable)."""

    def test_travel_planning_with_budget(self, realistic_catalog):
        result = classify_query(
            "Plan a trip to Paris: find flights, search for hotels near the "
            "center, look up restaurants with good reviews, and optimize "
            "everything within a budget of $2000 maximum",
            realistic_catalog,
        )
        assert result['level'] >= 1, f"Expected SC(>=1), got SC({result['level']})"

    def test_multi_constraint_optimization(self, realistic_catalog):
        result = classify_query(
            "Find the cheapest flights to Rome under $500, search for hotels "
            "between $80 and $120 per night near the Colosseum, find "
            "vegetarian restaurants with at least 4 stars, and optimize the "
            "total trip cost to stay within a $1500 budget",
            realistic_catalog,
        )
        assert result['level'] >= 1, f"Expected SC(>=1), got SC({result['level']})"

    def test_spanish_constrained(self, realistic_catalog):
        result = classify_query(
            "Buscar vuelos baratos a Lisboa, hoteles dentro de un presupuesto "
            "de 100 euros por noche, restaurantes con buenas reviews, y "
            "optimizar el coste total del viaje al mínimo posible",
            realistic_catalog,
        )
        assert result['level'] >= 1, f"Expected SC(>=1), got SC({result['level']})"


# ---------------------------------------------------------------------------
# SC(3): Entangled — cross-reference, analysis chains, agent delegation
# ---------------------------------------------------------------------------

class TestSC3Entangled:
    """SC(3) queries should classify at level >= 2 (preferably 3)."""

    def test_cross_reference_and_predict(self, realistic_catalog):
        """REGRESSION TEST: cross-reference + predictive model = SC(3)."""
        result = classify_query(
            "Analyze market trends by searching the web, cross-reference "
            "with social media sentiment analysis, correlate the data "
            "patterns, and build a predictive model from the combined insights",
            realistic_catalog,
        )
        assert result['level'] >= 2, f"Expected SC(>=2), got SC({result['level']})"

    def test_cross_reference_with_sequential_markers(self, realistic_catalog):
        """REGRESSION TEST: sequential markers must NOT force SC(1)
        when cross-reference and analysis chains are present."""
        result = classify_query(
            "First search for climate data, then analyze temperature trends, "
            "next cross-reference with economic indicators, and finally "
            "build a predictive model from the synthesized data",
            realistic_catalog,
        )
        assert result['level'] >= 2, (
            f"REGRESSION: SC(3) query with sequential markers classified as "
            f"SC({result['level']}). The pipeline shortcut should NOT fire "
            f"when cross-reference/analysis indicators are present."
        )

    def test_entangled_spanish(self, realistic_catalog):
        """REGRESSION TEST: Spanish cross-reference query."""
        result = classify_query(
            "Analizar tendencias del mercado buscando en la web, cruzar "
            "con análisis de sentimiento de redes sociales, correlacionar "
            "los datos entre fuentes, y construir un modelo predictivo "
            "con los insights combinados",
            realistic_catalog,
        )
        assert result['level'] >= 2, f"Expected SC(>=2), got SC({result['level']})"

    def test_complex_multi_step_analysis(self, realistic_catalog):
        """Full analysis pipeline requiring agent-level coordination."""
        result = classify_query(
            "Search for competitor pricing data across multiple markets, "
            "analyze sentiment from customer reviews and social media, "
            "cross-reference pricing trends with sentiment shifts, "
            "build a forecast model, and generate a strategic report "
            "integrating all findings with budget optimization",
            realistic_catalog,
        )
        assert result['level'] >= 2, f"Expected SC(>=2), got SC({result['level']})"


# ---------------------------------------------------------------------------
# Feature extraction sanity checks
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    """Verify the 17 features are computed correctly."""

    def test_all_17_features_present(self, realistic_catalog):
        features = extract_query_features(
            "Search for flights and hotels", realistic_catalog
        )
        for fname in FEATURE_NAMES:
            assert fname in features, f"Missing feature: {fname}"
        assert len(features) == 17

    def test_features_are_numeric(self, realistic_catalog):
        features = extract_query_features(
            "Analyze trends and build a model", realistic_catalog
        )
        for fname, val in features.items():
            assert isinstance(val, (int, float)), (
                f"Feature {fname} is {type(val)}, expected numeric"
            )

    def test_simple_query_low_complexity(self, realistic_catalog):
        features = extract_query_features(
            "What is the weather?", realistic_catalog
        )
        assert features['num_constraints'] == 0
        assert features['multi_tool_tasks'] == 0

    def test_constrained_query_has_constraints(self, realistic_catalog):
        features = extract_query_features(
            "Find flights under $500 within a budget of $2000 maximum, "
            "at least 3 star hotels",
            realistic_catalog,
        )
        assert features['num_constraints'] >= 2, (
            f"Expected >= 2 constraints, got {features['num_constraints']}"
        )


# ---------------------------------------------------------------------------
# Classification pipeline integrity
# ---------------------------------------------------------------------------

class TestClassificationIntegrity:
    """Verify the classification pipeline returns expected structure."""

    def test_returns_expected_keys(self, realistic_catalog):
        result = classify_query("What is the weather?", realistic_catalog)
        for key in ('level', 'confidence', 'evidence', 'phase'):
            assert key in result, f"Missing key: {key}"

    def test_level_always_in_range(self, realistic_catalog):
        queries = [
            "What is 2+2?",
            "Search and then summarize",
            "Plan trip with flights, hotels, restaurants under budget",
            "Cross-reference trends with sentiment and build predictive model",
        ]
        for query in queries:
            result = classify_query(query, realistic_catalog)
            assert 0 <= result['level'] <= 3, (
                f"Level {result['level']} out of range for: {query[:50]}"
            )

    def test_confidence_is_valid(self, realistic_catalog):
        result = classify_query(
            "Search for flights to Paris", realistic_catalog
        )
        assert result['confidence'] in ('high', 'medium', 'low')

    def test_evidence_contains_features(self, realistic_catalog):
        result = classify_query(
            "Analyze market trends", realistic_catalog
        )
        assert 'features' in result['evidence']
        assert 'patterns' in result['evidence']
