"""Tests for SC classification pipeline."""

from sc_router import ToolCatalog, Tool, classify_query


class TestClassifyQuery:
    def test_sc0_simple_question(self, sample_catalog):
        result = classify_query("What is the weather in Madrid?", sample_catalog)
        assert result['level'] == 0
        assert result['confidence'] in ('high', 'medium')

    def test_sc0_single_tool(self, sample_catalog):
        result = classify_query("Calculate 15 * 37", sample_catalog)
        assert result['level'] == 0

    def test_sc1_sequential(self, sample_catalog):
        result = classify_query(
            "First search the web for Python tutorials, then summarize the results",
            sample_catalog,
        )
        assert result['level'] in (0, 1)  # should detect pipeline

    def test_sc2_constrained(self, sample_catalog):
        result = classify_query(
            "Plan a trip to Paris: find flights, search for hotels near the center, "
            "look up restaurants with good reviews, and optimize everything "
            "within a budget of $2000 maximum",
            sample_catalog,
        )
        assert result['level'] >= 1  # at least decomposable

    def test_sc3_entangled(self, sample_catalog):
        result = classify_query(
            "Analyze market trends by searching the web, cross-reference with "
            "social media sentiment analysis, correlate the data patterns, "
            "and build a predictive model from the combined insights",
            sample_catalog,
        )
        assert result['level'] >= 2  # at least complex

    def test_returns_expected_keys(self, sample_catalog):
        result = classify_query("What is the weather?", sample_catalog)
        assert 'level' in result
        assert 'confidence' in result
        assert 'evidence' in result
        assert 'phase' in result

    def test_level_in_range(self, sample_catalog):
        for query in [
            "What is 2+2?",
            "Search and summarize",
            "Plan a complex trip with budget",
            "Analyze trends and cross-reference sentiment and build model",
        ]:
            result = classify_query(query, sample_catalog)
            assert 0 <= result['level'] <= 3
