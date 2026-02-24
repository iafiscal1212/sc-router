"""Tests for RemoteAgent and AgentRegistry."""

import pytest
from sc_router.agent import AgentStatus, RemoteAgent, AgentRegistry
from sc_router.catalog import Tool, ToolCatalog


def _make_tool(name="test-tool", tags=None):
    return Tool(
        name=name,
        description=f"Test tool {name}",
        input_types={"query"},
        output_types={"result"},
        capability_tags=set(tags or ["test"]),
    )


def _make_agent(agent_id="agent-1", url="http://localhost:8081", tool=None):
    if tool is None:
        tool = _make_tool(name=f"{agent_id}-tool")
    return RemoteAgent(id=agent_id, url=url, tool=tool)


class TestRemoteAgent:
    def test_creation(self):
        tool = _make_tool()
        agent = RemoteAgent(id="a1", url="http://host:8081", tool=tool)
        assert agent.id == "a1"
        assert agent.url == "http://host:8081"
        assert agent.tool is tool
        assert agent.status == AgentStatus.UNKNOWN

    def test_endpoint(self):
        agent = _make_agent(url="http://host:8081")
        assert agent.endpoint == "http://host:8081/execute"

    def test_endpoint_trailing_slash(self):
        agent = _make_agent(url="http://host:8081/")
        assert agent.endpoint == "http://host:8081/execute"

    def test_health_endpoint(self):
        agent = _make_agent(url="http://host:8081")
        assert agent.health_endpoint == "http://host:8081/health"

    def test_is_healthy_unknown(self):
        agent = _make_agent()
        assert not agent.is_healthy

    def test_is_healthy_statuses(self):
        agent = _make_agent()
        agent.status = AgentStatus.HEALTHY
        assert agent.is_healthy
        agent.status = AgentStatus.HALF_OPEN
        assert agent.is_healthy
        agent.status = AgentStatus.UNHEALTHY
        assert not agent.is_healthy

    def test_metadata(self):
        agent = RemoteAgent(
            id="a1", url="http://host:8081",
            tool=_make_tool(), metadata={"env": "prod"})
        assert agent.metadata["env"] == "prod"


class TestAgentRegistry:
    def test_register_and_lookup(self):
        reg = AgentRegistry()
        agent = _make_agent("search", tool=_make_tool("search"))
        reg.register(agent)

        assert reg.size == 1
        assert reg.get_agent("search") is agent
        assert reg.catalog.size == 1
        assert reg.catalog.get("search") is not None

    def test_unregister(self):
        reg = AgentRegistry()
        reg.register(_make_agent("a1", tool=_make_tool("t1")))
        reg.unregister("a1")
        assert reg.size == 0
        assert reg.catalog.size == 0

    def test_reverse_lookup(self):
        reg = AgentRegistry()
        agent = _make_agent("search-agent", tool=_make_tool("search"))
        reg.register(agent)

        found = reg.agent_for_tool("search")
        assert found is agent
        assert reg.agent_for_tool("nonexistent") is None

    def test_catalog_works_with_route(self):
        """registry.catalog should work with the standard route() function."""
        from sc_router import route

        reg = AgentRegistry()
        reg.register(_make_agent("weather-agent", tool=Tool(
            name="weather",
            description="Get weather forecast",
            input_types={"location"},
            output_types={"weather_data"},
            capability_tags={"weather", "forecast", "temperature"},
        )))

        result = route("What's the weather in Madrid?", reg.catalog)
        assert result.sc_level == 0
        assert result.tool_assignments[0].tool == "weather"

    def test_healthy_agents(self):
        reg = AgentRegistry()
        a1 = _make_agent("a1", tool=_make_tool("t1"))
        a2 = _make_agent("a2", tool=_make_tool("t2"))
        reg.register(a1)
        reg.register(a2)

        # Initially UNKNOWN → not healthy
        assert len(reg.healthy_agents()) == 0

        reg.set_status("a1", AgentStatus.HEALTHY)
        assert len(reg.healthy_agents()) == 1

        reg.set_status("a2", AgentStatus.HEALTHY)
        assert len(reg.healthy_agents()) == 2

        reg.set_status("a1", AgentStatus.UNHEALTHY)
        assert len(reg.healthy_agents()) == 1

    def test_healthy_catalog(self):
        reg = AgentRegistry()
        a1 = _make_agent("a1", tool=_make_tool("t1"))
        a2 = _make_agent("a2", tool=_make_tool("t2"))
        reg.register(a1)
        reg.register(a2)
        reg.set_status("a1", AgentStatus.HEALTHY)

        hc = reg.healthy_catalog()
        assert hc.size == 1
        assert hc.get("t1") is not None
        assert hc.get("t2") is None

    def test_set_status(self):
        reg = AgentRegistry()
        agent = _make_agent("a1", tool=_make_tool("t1"))
        reg.register(agent)

        reg.set_status("a1", AgentStatus.HEALTHY)
        assert agent.status == AgentStatus.HEALTHY

    def test_to_dict(self):
        reg = AgentRegistry()
        reg.register(_make_agent("a1", url="http://h:80", tool=_make_tool("t1", tags=["web"])))
        data = reg.to_dict()
        assert len(data) == 1
        assert data[0]['id'] == "a1"
        assert data[0]['url'] == "http://h:80"
        assert data[0]['tool']['name'] == "t1"
        assert "web" in data[0]['tool']['capability_tags']

    def test_agents_property(self):
        reg = AgentRegistry()
        reg.register(_make_agent("a1", tool=_make_tool("t1")))
        reg.register(_make_agent("a2", tool=_make_tool("t2")))
        assert len(reg.agents) == 2

    def test_multiple_registers(self):
        """Registering agents with different IDs should accumulate."""
        reg = AgentRegistry()
        for i in range(5):
            reg.register(_make_agent(f"a{i}", tool=_make_tool(f"t{i}")))
        assert reg.size == 5
        assert reg.catalog.size == 5
