"""YAML configuration loader for distributed sc-router.

Loads a declarative YAML config and builds an AgentRegistry ready to use.

Uses only stdlib if pyyaml is not available (falls back to a minimal parser
for simple configs, but pyyaml is recommended).
"""

from typing import Any, Dict, Optional

from .agent import AgentRegistry, RemoteAgent
from .catalog import Tool


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file. Requires pyyaml."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "pyyaml is required for YAML config loading. "
            "Install it with: pip install sc-router[gateway]"
        )
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a raw config dict.

    Expected structure:
        agents:
          - id: <str>
            url: <str>
            tool:
              name: <str>
              description: <str>
              capability_tags: [<str>, ...]
              input_types: [<str>, ...]
              output_types: [<str>, ...]
              cost: <float>  # optional, default 1.0
            metadata: {}  # optional

        health:  # optional
          failure_threshold: <int>  # default 3
          recovery_timeout_s: <float>  # default 30
          check_timeout_s: <float>  # default 5

        gateway:  # optional
          host: <str>  # default 0.0.0.0
          port: <int>  # default 8080
    """
    if not isinstance(raw, dict):
        raise ValueError("Config must be a dict")

    agents_raw = raw.get('agents', [])
    if not isinstance(agents_raw, list):
        raise ValueError("'agents' must be a list")

    agents = []
    for i, entry in enumerate(agents_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Agent entry {i} must be a dict")
        if 'id' not in entry:
            raise ValueError(f"Agent entry {i} missing 'id'")
        if 'url' not in entry:
            raise ValueError(f"Agent entry {i} missing 'url'")
        if 'tool' not in entry or not isinstance(entry['tool'], dict):
            raise ValueError(f"Agent entry {i} missing 'tool' dict")

        tool_raw = entry['tool']
        if 'name' not in tool_raw:
            raise ValueError(f"Agent {entry['id']} tool missing 'name'")
        if 'description' not in tool_raw:
            raise ValueError(f"Agent {entry['id']} tool missing 'description'")

        agents.append({
            'id': str(entry['id']),
            'url': str(entry['url']),
            'tool': {
                'name': str(tool_raw['name']),
                'description': str(tool_raw['description']),
                'capability_tags': set(tool_raw.get('capability_tags', [])),
                'input_types': set(tool_raw.get('input_types', [])),
                'output_types': set(tool_raw.get('output_types', [])),
                'cost': float(tool_raw.get('cost', 1.0)),
            },
            'metadata': dict(entry.get('metadata', {})),
        })

    health = raw.get('health', {})
    gateway = raw.get('gateway', {})

    return {
        'agents': agents,
        'health': {
            'failure_threshold': int(health.get('failure_threshold', 3)),
            'recovery_timeout_s': float(health.get('recovery_timeout_s', 30)),
            'check_timeout_s': float(health.get('check_timeout_s', 5)),
        },
        'gateway': {
            'host': str(gateway.get('host', '0.0.0.0')),
            'port': int(gateway.get('port', 8080)),
        },
    }


def build_registry(config: Dict[str, Any]) -> AgentRegistry:
    """Build an AgentRegistry from a parsed config dict.

    Args:
        config: Either a raw config dict or an already-parsed one.
                If it has a top-level 'agents' key with list of dicts
                containing 'tool' dicts with 'capability_tags' as lists,
                it will be parsed first.

    Returns:
        A fully populated AgentRegistry.
    """
    # Auto-parse if needed
    if config.get('agents') and isinstance(config['agents'], list):
        first = config['agents'][0] if config['agents'] else {}
        tool_data = first.get('tool', {})
        if isinstance(tool_data.get('capability_tags'), list):
            config = parse_config(config)

    registry = AgentRegistry()

    for agent_cfg in config.get('agents', []):
        tool_cfg = agent_cfg['tool']
        tool = Tool(
            name=tool_cfg['name'],
            description=tool_cfg['description'],
            input_types=set(tool_cfg.get('input_types', set())),
            output_types=set(tool_cfg.get('output_types', set())),
            capability_tags=set(tool_cfg.get('capability_tags', set())),
            cost=tool_cfg.get('cost', 1.0),
        )
        agent = RemoteAgent(
            id=agent_cfg['id'],
            url=agent_cfg['url'],
            tool=tool,
            metadata=agent_cfg.get('metadata', {}),
        )
        registry.register(agent)

    return registry


def load_config(path: str) -> Dict[str, Any]:
    """Load and parse a YAML config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed and validated config dict.
    """
    raw = _load_yaml(path)
    return parse_config(raw)


def load_registry(path: str) -> AgentRegistry:
    """Load a YAML config and build an AgentRegistry in one step.

    Args:
        path: Path to the YAML config file.

    Returns:
        A fully populated AgentRegistry.
    """
    config = load_config(path)
    return build_registry(config)
