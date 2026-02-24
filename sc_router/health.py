"""Health checks and circuit breaker for remote agents.

HealthChecker pings each agent's /health endpoint.
Circuit breaker: N failures → UNHEALTHY → wait recovery_timeout → HALF_OPEN → retry.

Uses only stdlib (urllib) by default. Uses httpx.AsyncClient if available.
"""

import asyncio
import time
from typing import Dict, Optional, TYPE_CHECKING

from .agent import AgentRegistry, AgentStatus, RemoteAgent

if TYPE_CHECKING:
    pass


class CircuitState:
    """Per-agent circuit breaker state."""

    __slots__ = ('failures', 'last_failure', 'last_success')

    def __init__(self) -> None:
        self.failures: int = 0
        self.last_failure: float = 0.0
        self.last_success: float = 0.0

    def record_success(self) -> None:
        self.failures = 0
        self.last_success = time.monotonic()

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure = time.monotonic()


class HealthChecker:
    """Health checker with circuit breaker for an AgentRegistry.

    Args:
        registry: The agent registry to monitor.
        failure_threshold: Consecutive failures before marking UNHEALTHY.
        recovery_timeout: Seconds to wait before trying HALF_OPEN.
        check_timeout: Seconds to wait for a single health check response.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        check_timeout: float = 5.0,
    ) -> None:
        self._registry = registry
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._check_timeout = check_timeout
        self._states: Dict[str, CircuitState] = {}

    def _get_state(self, agent_id: str) -> CircuitState:
        if agent_id not in self._states:
            self._states[agent_id] = CircuitState()
        return self._states[agent_id]

    def _should_attempt(self, agent: RemoteAgent) -> bool:
        """Decide if we should attempt a health check for this agent."""
        state = self._get_state(agent.id)
        if agent.status == AgentStatus.UNHEALTHY:
            elapsed = time.monotonic() - state.last_failure
            if elapsed >= self._recovery_timeout:
                self._registry.set_status(agent.id, AgentStatus.HALF_OPEN)
                return True
            return False
        return True

    async def check_agent(self, agent: RemoteAgent) -> bool:
        """Check a single agent's health. Returns True if healthy."""
        if not self._should_attempt(agent):
            return False

        state = self._get_state(agent.id)
        healthy = await self._ping(agent.health_endpoint)

        if healthy:
            state.record_success()
            self._registry.set_status(agent.id, AgentStatus.HEALTHY)
            return True
        else:
            state.record_failure()
            if state.failures >= self._failure_threshold:
                self._registry.set_status(agent.id, AgentStatus.UNHEALTHY)
            return False

    async def check_all(self) -> Dict[str, bool]:
        """Check all agents in parallel. Returns {agent_id: is_healthy}."""
        agents = self._registry.agents
        if not agents:
            return {}

        tasks = [self.check_agent(agent) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            agent.id: (r is True)
            for agent, r in zip(agents, results)
        }

    async def _ping(self, url: str) -> bool:
        """Ping a health endpoint. Uses httpx if available, else urllib."""
        try:
            return await self._ping_httpx(url)
        except ImportError:
            return await self._ping_urllib(url)

    async def _ping_httpx(self, url: str) -> bool:
        """Health check via httpx.AsyncClient."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=self._check_timeout) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception:
            return False

    async def _ping_urllib(self, url: str) -> bool:
        """Health check via urllib (stdlib fallback, run in executor)."""
        import urllib.request
        import urllib.error

        def _do_request() -> bool:
            try:
                req = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(req, timeout=self._check_timeout) as resp:
                    return resp.status == 200
            except Exception:
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _do_request)

    def get_status_summary(self) -> Dict[str, Dict]:
        """Return status summary for all agents."""
        result = {}
        for agent in self._registry.agents:
            state = self._get_state(agent.id)
            result[agent.id] = {
                'status': agent.status.value,
                'consecutive_failures': state.failures,
                'last_success': state.last_success,
                'last_failure': state.last_failure,
            }
        return result
