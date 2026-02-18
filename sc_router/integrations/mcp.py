"""MCP (Model Context Protocol) integration adapter.

Converts MCP tool definitions to SC-Router Tool objects.
"""

import re
from typing import Any, Dict, List, Optional

from ..catalog import Tool, ToolCatalog
from ..router import RoutingResult, route


def from_mcp_tools(tools: List[Dict[str, Any]]) -> List[Tool]:
    """Convert MCP tool definitions to SC-Router Tools.

    MCP tools follow the format:
    {
        "name": "tool_name",
        "description": "What the tool does",
        "inputSchema": {"type": "object", "properties": {...}}
    }
    """
    result = []
    for mcp_tool in tools:
        name = mcp_tool.get('name', 'unknown')
        description = mcp_tool.get('description', '')

        # Extract input types from inputSchema
        input_types = set()
        schema = mcp_tool.get('inputSchema', {})
        for prop_name in schema.get('properties', {}):
            input_types.add(prop_name)

        # Generate capability tags
        words = re.findall(r'[a-z]{3,}', description.lower())
        stop = {'the', 'and', 'for', 'with', 'from', 'that', 'this'}
        tags = {w for w in words if w not in stop}
        tags.add(name.lower().replace('_', '').replace('-', ''))

        result.append(Tool(
            name=name,
            description=description,
            input_types=input_types,
            output_types=set(),
            capability_tags=tags,
        ))

    return result


def build_catalog_from_mcp(tools: List[Dict]) -> ToolCatalog:
    """Build a ToolCatalog from MCP tool definitions."""
    catalog = ToolCatalog()
    for tool in from_mcp_tools(tools):
        catalog.register(tool)
    return catalog


def sc_route_mcp(
    query: str,
    tools: List[Dict],
    catalog: Optional[ToolCatalog] = None,
) -> RoutingResult:
    """Route a query using SC-Router with MCP tool definitions."""
    if catalog is None:
        catalog = build_catalog_from_mcp(tools)
    return route(query, catalog)
