"""Tests for YAML config loading and registry building."""

import os
import tempfile
import pytest

from sc_router.config import parse_config, build_registry


SAMPLE_CONFIG = {
    'agents': [
        {
            'id': 'search-agent',
            'url': 'http://search:8081',
            'tool': {
                'name': 'search',
                'description': 'Search the web',
                'capability_tags': ['search', 'web'],
                'input_types': ['query'],
                'output_types': ['search_results'],
            },
        },
        {
            'id': 'weather-agent',
            'url': 'http://weather:8082',
            'tool': {
                'name': 'weather',
                'description': 'Get weather forecast',
                'capability_tags': ['weather', 'forecast'],
                'input_types': ['location'],
                'output_types': ['weather_data'],
                'cost': 0.5,
            },
            'metadata': {'region': 'eu'},
        },
    ],
    'health': {
        'failure_threshold': 5,
        'recovery_timeout_s': 60,
    },
    'gateway': {
        'host': '127.0.0.1',
        'port': 9090,
    },
}


class TestParseConfig:
    def test_basic_parsing(self):
        config = parse_config(SAMPLE_CONFIG)
        assert len(config['agents']) == 2
        assert config['agents'][0]['id'] == 'search-agent'
        assert config['agents'][0]['url'] == 'http://search:8081'

    def test_tool_fields(self):
        config = parse_config(SAMPLE_CONFIG)
        tool = config['agents'][0]['tool']
        assert tool['name'] == 'search'
        assert 'search' in tool['capability_tags']
        assert 'query' in tool['input_types']
        assert 'search_results' in tool['output_types']

    def test_tool_cost(self):
        config = parse_config(SAMPLE_CONFIG)
        assert config['agents'][1]['tool']['cost'] == 0.5
        assert config['agents'][0]['tool']['cost'] == 1.0  # default

    def test_health_config(self):
        config = parse_config(SAMPLE_CONFIG)
        assert config['health']['failure_threshold'] == 5
        assert config['health']['recovery_timeout_s'] == 60

    def test_gateway_config(self):
        config = parse_config(SAMPLE_CONFIG)
        assert config['gateway']['host'] == '127.0.0.1'
        assert config['gateway']['port'] == 9090

    def test_defaults(self):
        config = parse_config({'agents': [{
            'id': 'a1', 'url': 'http://a:80',
            'tool': {'name': 't1', 'description': 'd1'},
        }]})
        assert config['health']['failure_threshold'] == 3
        assert config['health']['recovery_timeout_s'] == 30
        assert config['gateway']['host'] == '0.0.0.0'
        assert config['gateway']['port'] == 8080

    def test_metadata(self):
        config = parse_config(SAMPLE_CONFIG)
        assert config['agents'][1]['metadata'] == {'region': 'eu'}
        assert config['agents'][0]['metadata'] == {}

    def test_invalid_not_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            parse_config("not a dict")

    def test_invalid_agents_not_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            parse_config({'agents': 'bad'})

    def test_missing_id(self):
        with pytest.raises(ValueError, match="missing 'id'"):
            parse_config({'agents': [{'url': 'x', 'tool': {'name': 'n', 'description': 'd'}}]})

    def test_missing_url(self):
        with pytest.raises(ValueError, match="missing 'url'"):
            parse_config({'agents': [{'id': 'a', 'tool': {'name': 'n', 'description': 'd'}}]})

    def test_missing_tool(self):
        with pytest.raises(ValueError, match="missing 'tool'"):
            parse_config({'agents': [{'id': 'a', 'url': 'x'}]})

    def test_missing_tool_name(self):
        with pytest.raises(ValueError, match="missing 'name'"):
            parse_config({'agents': [{'id': 'a', 'url': 'x',
                                       'tool': {'description': 'd'}}]})

    def test_missing_tool_description(self):
        with pytest.raises(ValueError, match="missing 'description'"):
            parse_config({'agents': [{'id': 'a', 'url': 'x',
                                       'tool': {'name': 'n'}}]})


class TestBuildRegistry:
    def test_builds_from_raw_config(self):
        registry = build_registry(SAMPLE_CONFIG)
        assert registry.size == 2
        assert registry.catalog.size == 2

    def test_builds_from_parsed_config(self):
        parsed = parse_config(SAMPLE_CONFIG)
        registry = build_registry(parsed)
        assert registry.size == 2

    def test_agent_lookup(self):
        registry = build_registry(SAMPLE_CONFIG)
        agent = registry.get_agent('search-agent')
        assert agent is not None
        assert agent.url == 'http://search:8081'
        assert agent.tool.name == 'search'

    def test_reverse_lookup(self):
        registry = build_registry(SAMPLE_CONFIG)
        agent = registry.agent_for_tool('weather')
        assert agent is not None
        assert agent.id == 'weather-agent'

    def test_tool_in_catalog(self):
        registry = build_registry(SAMPLE_CONFIG)
        tool = registry.catalog.get('search')
        assert tool is not None
        assert tool.description == 'Search the web'
        assert 'web' in tool.capability_tags

    def test_empty_config(self):
        registry = build_registry({'agents': []})
        assert registry.size == 0

    def test_metadata_preserved(self):
        registry = build_registry(SAMPLE_CONFIG)
        agent = registry.get_agent('weather-agent')
        assert agent.metadata == {'region': 'eu'}


class TestLoadYaml:
    def test_load_yaml_file(self):
        """Test loading from a YAML file (requires pyyaml)."""
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")

        from sc_router.config import load_config, load_registry

        yaml_content = """
agents:
  - id: test-agent
    url: http://test:8080
    tool:
      name: test
      description: "Test tool"
      capability_tags:
        - test
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            config = load_config(path)
            assert len(config['agents']) == 1
            assert config['agents'][0]['id'] == 'test-agent'

            registry = load_registry(path)
            assert registry.size == 1
        finally:
            os.unlink(path)
