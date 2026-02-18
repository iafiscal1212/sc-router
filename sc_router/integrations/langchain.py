"""LangChain integration adapter.

Converts LangChain tools to SC-Router Tool objects and provides
an SC-aware router that wraps LangChain's tool execution.
"""

from typing import Any, List, Optional

from ..catalog import Tool, ToolCatalog
from ..router import RoutingResult, route


def from_langchain_tools(tools: List[Any]) -> List[Tool]:
    """Convert LangChain BaseTool instances to SC-Router Tools.

    Args:
        tools: List of LangChain BaseTool instances.

    Returns:
        List of SC-Router Tool objects.
    """
    import re
    result = []
    for lc_tool in tools:
        name = getattr(lc_tool, 'name', str(lc_tool))
        description = getattr(lc_tool, 'description', '')

        # Extract capability tags from description
        words = re.findall(r'[a-z]{3,}', description.lower())
        stop = {'the', 'and', 'for', 'with', 'from', 'that', 'this'}
        tags = {w for w in words if w not in stop}
        tags.add(name.lower().replace('_', '').replace('-', ''))

        # Extract input/output from args_schema if available
        input_types = set()
        if hasattr(lc_tool, 'args_schema') and lc_tool.args_schema:
            schema = lc_tool.args_schema
            if hasattr(schema, 'schema'):
                for prop_name in schema.schema().get('properties', {}):
                    input_types.add(prop_name)

        result.append(Tool(
            name=name,
            description=description,
            input_types=input_types,
            output_types=set(),
            capability_tags=tags,
        ))

    return result


def build_catalog_from_langchain(tools: List[Any]) -> ToolCatalog:
    """Build a ToolCatalog from LangChain tools."""
    catalog = ToolCatalog()
    for tool in from_langchain_tools(tools):
        catalog.register(tool)
    return catalog


def sc_route_langchain(
    query: str,
    tools: List[Any],
    catalog: Optional[ToolCatalog] = None,
) -> RoutingResult:
    """Route a query using SC-Router with LangChain tools.

    Args:
        query: The query to route.
        tools: LangChain BaseTool instances.
        catalog: Optional pre-built catalog (avoids rebuilding).

    Returns:
        RoutingResult with tool assignments.
    """
    if catalog is None:
        catalog = build_catalog_from_langchain(tools)
    return route(query, catalog)
