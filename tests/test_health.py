"""Tests for HealthChecker and circuit breaker."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from sc_router.agent import AgentStatus, RemoteAgent, AgentRegistry
from sc_router.catalog import Tool
from sc_router.health import HealthChecker, CircuitState


def _make_registry(*agents_data):
    """Build a registry with agents. agents_data = list of (id, url) tuples."""
    reg = AgentRegistry()
    for agent_id, url in agents_data:
        tool = Tool(
            name=f"{agent_id}-tool",
            description=f"Tool for {agent_id}",
            input_types={"query"},
            output_types={"result"},
            capability_tags={"test"},
        )
        reg.register(RemoteAgent(id=agent_id, url=url, tool=tool))
    return reg


class TestCircuitState:
    def test_initial_state(self):
        state = CircuitState()
        assert state.failures == 0
        assert state.last_failure == 0.0
        assert state.last_success == 0.0

    def test_record_success(self):
        state = CircuitState()
        state.record_failure()
        state.record_failure()
        assert state.failures == 2
        state.record_success()
        assert state.failures == 0
        assert state.last_success > 0

    def test_record_failure(self):
        state = CircuitState()
        state.record_failure()
        assert state.failures == 1
        assert state.last_failure > 0


class TestHealthChecker:
    @pytest.mark.asyncio
    async def test_healthy_agent(self):
        reg = _make_registry(("a1", "http://localhost:8081"))
        checker = HealthChecker(reg, failure_threshold=3)

        with patch.object(checker, '_ping', new_callable=AsyncMock, return_value=True):
            result = await checker.check_agent(reg.get_agent("a1"))
            assert result is True
            assert reg.get_agent("a1").status == AgentStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_after_threshold(self):
        reg = _make_registry(("a1", "http://localhost:8081"))
        checker = HealthChecker(reg, failure_threshold=3)

        with patch.object(checker, '_ping', new_callable=AsyncMock, return_value=False):
            # 3 failures → UNHEALTHY
            for _ in range(3):
                await checker.check_agent(reg.get_agent("a1"))
            assert reg.get_agent("a1").status == AgentStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_not_unhealthy_before_threshold(self):
        reg = _make_registry(("a1", "http://localhost:8081"))
        checker = HealthChecker(reg, failure_threshold=3)

        with patch.object(checker, '_ping', new_callable=AsyncMock, return_value=False):
            await checker.check_agent(reg.get_agent("a1"))
            await checker.check_agent(reg.get_agent("a1"))
            # Only 2 failures — not yet UNHEALTHY
            assert reg.get_agent("a1").status != AgentStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_recovery_to_half_open(self):
        """After recovery_timeout, UNHEALTHY agent becomes HALF_OPEN."""
        reg = _make_registry(("a1", "http://localhost:8081"))
        checker = HealthChecker(reg, failure_threshold=1, recovery_timeout=0.0)

        with patch.object(checker, '_ping', new_callable=AsyncMock, return_value=False):
            await checker.check_agent(reg.get_agent("a1"))
            assert reg.get_agent("a1").status == AgentStatus.UNHEALTHY

        # Recovery timeout = 0, so immediately allows retry
        with patch.object(checker, '_ping', new_callable=AsyncMock, return_value=True):
            result = await checker.check_agent(reg.get_agent("a1"))
            assert result is True
            assert reg.get_agent("a1").status == AgentStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_skips_check_before_timeout(self):
        """UNHEALTHY agent with recovery_timeout > 0 should be skipped."""
        reg = _make_registry(("a1", "http://localhost:8081"))
        checker = HealthChecker(reg, failure_threshold=1, recovery_timeout=9999)

        with patch.object(checker, '_ping', new_callable=AsyncMock, return_value=False):
            await checker.check_agent(reg.get_agent("a1"))
            assert reg.get_agent("a1").status == AgentStatus.UNHEALTHY

        # Ping should NOT be called because recovery_timeout hasn't elapsed
        mock_ping = AsyncMock(return_value=True)
        with patch.object(checker, '_ping', mock_ping):
            result = await checker.check_agent(reg.get_agent("a1"))
            assert result is False
            mock_ping.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_all(self):
        reg = _make_registry(
            ("a1", "http://localhost:8081"),
            ("a2", "http://localhost:8082"),
        )
        checker = HealthChecker(reg, failure_threshold=3)

        with patch.object(checker, '_ping', new_callable=AsyncMock, return_value=True):
            results = await checker.check_all()
            assert results == {"a1": True, "a2": True}

    @pytest.mark.asyncio
    async def test_check_all_mixed(self):
        reg = _make_registry(
            ("a1", "http://localhost:8081"),
            ("a2", "http://localhost:8082"),
        )
        checker = HealthChecker(reg, failure_threshold=1)

        call_count = 0
        async def _alternating_ping(url):
            nonlocal call_count
            call_count += 1
            return "8081" in url  # a1 healthy, a2 not

        with patch.object(checker, '_ping', side_effect=_alternating_ping):
            results = await checker.check_all()
            assert results["a1"] is True
            assert results["a2"] is False

    @pytest.mark.asyncio
    async def test_check_all_empty(self):
        reg = AgentRegistry()
        checker = HealthChecker(reg)
        results = await checker.check_all()
        assert results == {}

    def test_get_status_summary(self):
        reg = _make_registry(("a1", "http://localhost:8081"))
        checker = HealthChecker(reg)

        summary = checker.get_status_summary()
        assert "a1" in summary
        assert summary["a1"]["status"] == "unknown"
        assert summary["a1"]["consecutive_failures"] == 0

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        reg = _make_registry(("a1", "http://localhost:8081"))
        checker = HealthChecker(reg, failure_threshold=3)

        with patch.object(checker, '_ping', new_callable=AsyncMock, return_value=False):
            await checker.check_agent(reg.get_agent("a1"))
            await checker.check_agent(reg.get_agent("a1"))

        state = checker._get_state("a1")
        assert state.failures == 2

        with patch.object(checker, '_ping', new_callable=AsyncMock, return_value=True):
            await checker.check_agent(reg.get_agent("a1"))

        assert state.failures == 0
