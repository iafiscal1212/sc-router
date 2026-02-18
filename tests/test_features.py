"""Tests for feature extraction."""

from sc_router import Tool, ToolCatalog, extract_query_features
from sc_router.features import _split_subtasks, _extract_keywords, FEATURE_NAMES


class TestSplitSubtasks:
    def test_single_sentence(self):
        result = _split_subtasks("What is the weather?")
        assert len(result) == 1

    def test_conjunction_split(self):
        result = _split_subtasks("Search for flights and book the cheapest one")
        assert len(result) >= 2

    def test_sequential_markers(self):
        result = _split_subtasks("First search for flights, then book the cheapest")
        assert len(result) >= 2

    def test_numbered_list(self):
        result = _split_subtasks("1. Search flights\n2. Compare prices\n3. Book cheapest")
        assert len(result) == 3

    def test_comma_list(self):
        result = _split_subtasks("flights, hotels, restaurants")
        assert len(result) == 3


class TestExtractKeywords:
    def test_filters_stop_words(self):
        kw = _extract_keywords("what is the weather in Madrid")
        assert "the" not in kw
        assert "weather" in kw
        assert "madrid" in kw

    def test_empty_string(self):
        kw = _extract_keywords("")
        assert len(kw) == 0


class TestExtractQueryFeatures:
    def test_returns_17_features(self, sample_catalog):
        features = extract_query_features("What is the weather?", sample_catalog)
        assert len(features) == 17
        for name in FEATURE_NAMES:
            assert name in features

    def test_simple_query_low_features(self, sample_catalog):
        features = extract_query_features("What is the weather in Madrid?", sample_catalog)
        assert features['num_candidate_tools'] <= 3
        assert features['multi_tool_tasks'] == 0

    def test_complex_query_high_features(self, sample_catalog):
        query = (
            "Search for flights to Paris, find hotels near the center, "
            "look up restaurants with good reviews, and optimize everything "
            "within a $2000 budget"
        )
        features = extract_query_features(query, sample_catalog)
        assert features['num_candidate_tools'] >= 3
        assert features['num_constraints'] >= 1

    def test_all_features_numeric(self, sample_catalog):
        features = extract_query_features("Calculate 2 + 2", sample_catalog)
        for name, val in features.items():
            assert isinstance(val, (int, float)), f"{name} is not numeric: {type(val)}"
