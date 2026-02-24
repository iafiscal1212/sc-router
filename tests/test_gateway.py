"""Tests for the gateway HTTP endpoints."""

import pytest

# Skip all tests if starlette is not installed
starlette = pytest.importorskip("starlette")
httpx_mod = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_router.agent import AgentStatus, RemoteAgent, AgentRegistry
from sc_router.catalog import Tool
from sc_router.gateway import create_app


def _make_app():
    """Build a gateway app with test agents."""
    registry = AgentRegistry()
    registry.register(RemoteAgent(
        id="search-agent",
        url="http://search:8081",
        tool=Tool(
            name="search", description="Search the web for information",
            input_types={"query"}, output_types={"search_results"},
            capability_tags={"search", "web", "find", "lookup", "information"},
        ),
        status=AgentStatus.HEALTHY,
    ))
    registry.register(RemoteAgent(
        id="weather-agent",
        url="http://weather:8082",
        tool=Tool(
            name="weather", description="Get weather forecast for a location",
            input_types={"location"}, output_types={"weather_data"},
            capability_tags={"weather", "forecast", "temperature"},
        ),
        status=AgentStatus.HEALTHY,
    ))

    app = create_app(registry=registry)
    return app


class TestRouteEndpoint:
    def test_classify_only(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.post('/route', json={
            'query': "What's the weather in Madrid?",
            'execute': False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data['sc_level'] == 0
        assert data['query'] == "What's the weather in Madrid?"
        assert len(data['tool_assignments']) >= 1
        assert 'trace' in data
        assert 'execution' not in data

    def test_missing_query(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.post('/route', json={})
        assert resp.status_code == 400
        assert 'error' in resp.json()

    def test_empty_query(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.post('/route', json={'query': ''})
        assert resp.status_code == 400

    def test_invalid_json(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.post('/route', content=b'not json',
                          headers={'content-type': 'application/json'})
        assert resp.status_code == 400


class TestHealthEndpoint:
    def test_health_all_healthy(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'healthy'
        assert data['agents_total'] == 2
        assert data['agents_healthy'] == 2

    def test_health_degraded(self):
        app = _make_app()
        # Make all agents unhealthy
        for agent in app.state.registry.agents:
            app.state.registry.set_status(agent.id, AgentStatus.UNHEALTHY)

        client = TestClient(app)
        resp = client.get('/health')
        data = resp.json()
        assert data['status'] == 'degraded'
        assert data['agents_healthy'] == 0

    def test_health_empty_registry(self):
        app = create_app(registry=AgentRegistry())
        client = TestClient(app)
        resp = client.get('/health')
        data = resp.json()
        assert data['status'] == 'healthy'
        assert data['agents_total'] == 0


class TestAgentsEndpoint:
    def test_list_agents(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get('/agents')
        assert resp.status_code == 200
        data = resp.json()
        assert data['count'] == 2
        assert len(data['agents']) == 2

        ids = {a['id'] for a in data['agents']}
        assert 'search-agent' in ids
        assert 'weather-agent' in ids

    def test_agent_details(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get('/agents')
        data = resp.json()

        search = [a for a in data['agents'] if a['id'] == 'search-agent'][0]
        assert search['url'] == 'http://search:8081'
        assert search['tool']['name'] == 'search'
        assert 'web' in search['tool']['capability_tags']

    def test_empty_agents(self):
        app = create_app(registry=AgentRegistry())
        client = TestClient(app)
        resp = client.get('/agents')
        data = resp.json()
        assert data['count'] == 0
        assert data['agents'] == []
