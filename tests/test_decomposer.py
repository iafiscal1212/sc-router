"""Tests for query decomposition."""

from sc_router import decompose, ToolCatalog, Tool


class TestDecompose:
    def test_single_task(self, sample_catalog):
        result = decompose("What is the weather?", sample_catalog)
        assert len(result.subtasks) == 1

    def test_sequential_decomposition(self, sample_catalog):
        result = decompose(
            "First search the web for Python tutorials, then summarize the results",
            sample_catalog,
        )
        assert len(result.subtasks) >= 2
        assert result.mode in ('sequential', 'single')

    def test_parallel_decomposition(self, sample_catalog):
        result = decompose(
            "Search for flights and also search for hotels at the same time",
            sample_catalog,
        )
        assert len(result.subtasks) >= 2

    def test_numbered_list(self, sample_catalog):
        result = decompose(
            "1. Search for flights\n2. Find hotels\n3. Look up restaurants",
            sample_catalog,
        )
        assert len(result.subtasks) == 3
        assert result.mode == 'sequential'

    def test_subtasks_have_tool_assignments(self, sample_catalog):
        result = decompose(
            "Search the web for news, then summarize the results",
            sample_catalog,
        )
        for st in result.subtasks:
            # Each subtask should have attempted tool matching
            assert isinstance(st.tools, list)

    def test_dependencies_sequential(self, sample_catalog):
        result = decompose(
            "1. Search flights\n2. Compare prices\n3. Book cheapest",
            sample_catalog,
        )
        if result.mode == 'sequential' and len(result.subtasks) >= 3:
            # Second subtask depends on first
            assert 0 in result.subtasks[1].depends_on
            # Third depends on second
            assert 1 in result.subtasks[2].depends_on

    def test_original_query_preserved(self, sample_catalog):
        query = "Search and summarize"
        result = decompose(query, sample_catalog)
        assert result.original_query == query
