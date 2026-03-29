"""Tests for ProfileManager (LLM EKG integration).

Requires llm-ekg >= 1.1.0 to run.
"""

import os
import tempfile

import pytest

from sc_router.profiles import ProfileManager, _MIN_RESPONSES


# Skip all tests if llm-ekg not installed
pytest.importorskip("llm_ekg", reason="llm-ekg not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_storage():
    """Temporary directory for profile storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def pm(tmp_storage):
    """ProfileManager with temporary storage."""
    return ProfileManager(storage_dir=tmp_storage)


# Sample responses for testing (diverse enough to produce real features)
_RESPONSES = {
    "good": (
        "The quarterly revenue increased by 15.3% year-over-year, reaching "
        "$42.7 million. Key drivers include: (1) expansion into European "
        "markets with 23% growth, (2) new enterprise contracts worth $8.2M, "
        "and (3) improved retention rates at 94.2%. Operating margins improved "
        "to 18.5% from 16.1% in Q3 2025."
    ),
    "hedgy": (
        "Perhaps the data might suggest some correlation, although it could "
        "possibly be coincidental. Generally speaking, the trends seem "
        "somewhat positive, but we should probably consider that maybe the "
        "sample size is relatively small and arguably insufficient."
    ),
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecordAndRetrieve:
    """Recording responses and retrieving profiles."""

    def test_record_returns_expected_keys(self, pm):
        result = pm.record("model-a", sc_level=1, response=_RESPONSES["good"])
        assert "anomaly_score" in result
        assert "global_score" in result
        assert "n_responses" in result

    def test_record_increments_count(self, pm):
        for i in range(3):
            result = pm.record(
                "model-a", sc_level=0,
                response=f"Response number {i} with some content.",
            )
        assert result["n_responses"] == 3

    def test_get_profile_returns_data(self, pm):
        pm.record("model-a", sc_level=2, response=_RESPONSES["good"])
        profile = pm.get_profile("model-a", sc_level=2)
        assert profile is not None
        assert profile["model"] == "model-a"
        assert profile["sc_level"] == 2
        assert profile["n_responses"] == 1

    def test_get_profile_nonexistent(self, pm):
        assert pm.get_profile("nonexistent", sc_level=0) is None


class TestBestModelSelection:
    """Selecting the best model based on EKG scores."""

    def test_best_model_with_enough_data(self, pm):
        # Record enough responses for two models
        for i in range(_MIN_RESPONSES + 1):
            pm.record(
                "model-good", sc_level=1,
                response=_RESPONSES["good"],
                response_time_s=0.5,
            )
            pm.record(
                "model-hedgy", sc_level=1,
                response=_RESPONSES["hedgy"],
                response_time_s=1.5,
            )

        best = pm.best_model(sc_level=1)
        assert best is not None
        # Both models have enough data; best_model returns one of them
        assert best in ("model-good", "model-hedgy")

    def test_best_model_with_candidates_filter(self, pm):
        for i in range(_MIN_RESPONSES + 1):
            pm.record("model-a", sc_level=0, response=_RESPONSES["good"])
            pm.record("model-b", sc_level=0, response=_RESPONSES["good"])

        best = pm.best_model(sc_level=0, candidates=["model-a"])
        assert best == "model-a"

    def test_insufficient_data_returns_none(self, pm):
        # Record fewer than _MIN_RESPONSES
        for i in range(_MIN_RESPONSES - 1):
            pm.record("model-a", sc_level=2, response=_RESPONSES["good"])
        assert pm.best_model(sc_level=2) is None

    def test_wrong_level_returns_none(self, pm):
        for i in range(_MIN_RESPONSES + 1):
            pm.record("model-a", sc_level=1, response=_RESPONSES["good"])
        assert pm.best_model(sc_level=3) is None


class TestPersistence:
    """Save and load profiles."""

    def test_save_creates_files(self, pm, tmp_storage):
        pm.record("model-x", sc_level=0, response=_RESPONSES["good"])
        pm.save()
        files = os.listdir(tmp_storage)
        assert len(files) == 1
        assert files[0].endswith(".json")

    def test_load_restores_profiles(self, tmp_storage):
        # Phase 1: create and save
        pm1 = ProfileManager(storage_dir=tmp_storage)
        for i in range(3):
            pm1.record("model-y", sc_level=1, response=_RESPONSES["good"])
        pm1.save()

        # Phase 2: load into fresh manager
        pm2 = ProfileManager(storage_dir=tmp_storage)
        profile = pm2.get_profile("model-y", sc_level=1)
        assert profile is not None
        assert profile["n_responses"] == 3

    def test_load_empty_dir(self, tmp_storage):
        # Should not raise
        pm = ProfileManager(storage_dir=tmp_storage)
        assert pm.all_profiles() == []


class TestAllProfiles:
    """Listing all profiles."""

    def test_all_profiles_multi_model_multi_level(self, pm):
        pm.record("m1", sc_level=0, response=_RESPONSES["good"])
        pm.record("m1", sc_level=1, response=_RESPONSES["good"])
        pm.record("m2", sc_level=0, response=_RESPONSES["hedgy"])

        profiles = pm.all_profiles()
        assert len(profiles) == 3

        models = {p["model"] for p in profiles}
        assert models == {"m1", "m2"}


class TestReset:
    """Resetting profiles."""

    def test_reset_all(self, pm):
        pm.record("m1", sc_level=0, response=_RESPONSES["good"])
        pm.record("m2", sc_level=1, response=_RESPONSES["good"])
        pm.reset()
        assert pm.all_profiles() == []

    def test_reset_by_model(self, pm):
        pm.record("m1", sc_level=0, response=_RESPONSES["good"])
        pm.record("m2", sc_level=0, response=_RESPONSES["good"])
        pm.reset(model="m1")
        profiles = pm.all_profiles()
        assert len(profiles) == 1
        assert profiles[0]["model"] == "m2"

    def test_reset_by_level(self, pm):
        pm.record("m1", sc_level=0, response=_RESPONSES["good"])
        pm.record("m1", sc_level=1, response=_RESPONSES["good"])
        pm.reset(sc_level=0)
        profiles = pm.all_profiles()
        assert len(profiles) == 1
        assert profiles[0]["sc_level"] == 1

    def test_reset_specific(self, pm):
        pm.record("m1", sc_level=0, response=_RESPONSES["good"])
        pm.record("m1", sc_level=1, response=_RESPONSES["good"])
        pm.reset(model="m1", sc_level=0)
        profiles = pm.all_profiles()
        assert len(profiles) == 1
        assert profiles[0]["sc_level"] == 1

    def test_reset_removes_files(self, pm, tmp_storage):
        pm.record("m1", sc_level=0, response=_RESPONSES["good"])
        pm.save()
        assert len(os.listdir(tmp_storage)) == 1
        pm.reset(model="m1")
        assert len(os.listdir(tmp_storage)) == 0


class TestRouterIntegration:
    """Verify ProfileManager works with route()."""

    def test_route_with_profile_manager(self, pm):
        from sc_router import Tool, ToolCatalog, route

        catalog = ToolCatalog()
        catalog.register(Tool(
            name="weather",
            description="Get weather forecast",
            input_types={"location"},
            output_types={"weather_data"},
            capability_tags={"weather", "forecast"},
        ))

        # Record enough data for a recommendation
        for i in range(_MIN_RESPONSES + 1):
            pm.record("best-model", sc_level=0, response=_RESPONSES["good"])

        result = route("What is the weather?", catalog, profile_manager=pm)
        assert result.sc_level == 0
        assert result.metadata.get("ekg_recommendation") == "best-model"

    def test_route_without_profile_manager(self):
        from sc_router import Tool, ToolCatalog, route

        catalog = ToolCatalog()
        catalog.register(Tool(
            name="calc",
            description="Calculate",
            input_types={"expression"},
            output_types={"number"},
            capability_tags={"math", "calculate"},
        ))

        result = route("Calculate 2+2", catalog)
        assert "ekg_recommendation" not in result.metadata
