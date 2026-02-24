"""Remote agent registry for distributed sc-router.

RemoteAgent uses composition (wraps a Tool), so ToolCatalog works unchanged.
AgentRegistry wraps a ToolCatalog and adds: URLs, health state, reverse lookup.
"""

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .catalog import Tool, ToolCatalog


class AgentStatus(enum.Enum):
    """Health status of a remote agent."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    HALF_OPEN = "half_open"
    UNKNOWN = "unknown"


@dataclass
class RemoteAgent:
    """A remote agent that exposes a Tool for catalog compatibility.

    Uses composition: contains a Tool instance so ToolCatalog can index it
    without any modifications.

    Attributes:
        id: Unique agent identifier.
        url: Base URL of the remote agent (e.g. http://search:8081).
        tool: The Tool this agent exposes to the catalog.
        status: Current health status.
        metadata: Arbitrary key-value metadata.
    """
    id: str
    url: str
    tool: Tool
    status: AgentStatus = AgentStatus.UNKNOWN
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        """Full execution endpoint URL."""
        return self.url.rstrip('/') + '/execute'

    @property
    def health_endpoint(self) -> str:
        """Health check endpoint URL."""
        return self.url.rstrip('/') + '/health'

    @property
    def is_healthy(self) -> bool:
        return self.status in (AgentStatus.HEALTHY, AgentStatus.HALF_OPEN)


class AgentRegistry:
    """Registry of remote agents backed by a ToolCatalog.

    Wraps a ToolCatalog so that ``route(query, registry.catalog)`` works
    identically to using a plain ToolCatalog.

    Adds:
      - Agent URL tracking
      - Health state per agent
      - Reverse lookup: tool name → agent
      - Filtering by health status
    """

    def __init__(self) -> None:
        self._catalog = ToolCatalog()
        self._agents: Dict[str, RemoteAgent] = {}
        self._tool_to_agent: Dict[str, str] = {}  # tool_name → agent_id

    # -- catalog access --

    @property
    def catalog(self) -> ToolCatalog:
        """The underlying ToolCatalog for use with ``route()``."""
        return self._catalog

    # -- agent management --

    def register(self, agent: RemoteAgent) -> None:
        """Register a remote agent and its tool in the catalog."""
        self._agents[agent.id] = agent
        self._catalog.register(agent.tool)
        self._tool_to_agent[agent.tool.name] = agent.id

    def unregister(self, agent_id: str) -> None:
        """Remove an agent and its tool from the registry."""
        agent = self._agents.pop(agent_id, None)
        if agent:
            self._catalog.unregister(agent.tool.name)
            self._tool_to_agent.pop(agent.tool.name, None)

    def get_agent(self, agent_id: str) -> Optional[RemoteAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def agent_for_tool(self, tool_name: str) -> Optional[RemoteAgent]:
        """Reverse lookup: get the agent that provides a given tool."""
        agent_id = self._tool_to_agent.get(tool_name)
        if agent_id:
            return self._agents.get(agent_id)
        return None

    @property
    def agents(self) -> List[RemoteAgent]:
        """All registered agents."""
        return list(self._agents.values())

    @property
    def size(self) -> int:
        return len(self._agents)

    def healthy_agents(self) -> List[RemoteAgent]:
        """Return only agents with HEALTHY or HALF_OPEN status."""
        return [a for a in self._agents.values() if a.is_healthy]

    def healthy_catalog(self) -> ToolCatalog:
        """Build a ToolCatalog containing only tools from healthy agents."""
        catalog = ToolCatalog()
        for agent in self.healthy_agents():
            catalog.register(agent.tool)
        return catalog

    def set_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update an agent's health status."""
        agent = self._agents.get(agent_id)
        if agent:
            agent.status = status

    def to_dict(self) -> List[Dict]:
        """Serialize all agents for API responses."""
        return [
            {
                'id': a.id,
                'url': a.url,
                'status': a.status.value,
                'tool': {
                    'name': a.tool.name,
                    'description': a.tool.description,
                    'capability_tags': sorted(a.tool.capability_tags),
                    'input_types': sorted(a.tool.input_types),
                    'output_types': sorted(a.tool.output_types),
                },
                'metadata': a.metadata,
            }
            for a in self._agents.values()
        ]
