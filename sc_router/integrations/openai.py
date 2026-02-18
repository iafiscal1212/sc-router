"""OpenAI function calling integration adapter.

Converts OpenAI function/tool definitions to SC-Router Tool objects.
"""

import re
from typing import Any, Dict, List, Optional

from ..catalog import Tool, ToolCatalog
from ..router import RoutingResult, route


def from_openai_functions(functions: List[Dict[str, Any]]) -> List[Tool]:
    """Convert OpenAI function definitions to SC-Router Tools.

    Accepts the format used in openai.chat.completions.create(tools=[...]).

    Args:
        functions: List of OpenAI function/tool dicts with 'type' and 'function' keys,
                   or direct function dicts with 'name', 'description', 'parameters'.
    """
    result = []
    for func_def in functions:
        # Handle both wrapped ({"type": "function", "function": {...}})
        # and unwrapped ({"name": ..., "description": ...}) formats
        if 'function' in func_def:
            func = func_def['function']
        else:
            func = func_def

        name = func.get('name', 'unknown')
        description = func.get('description', '')

        # Extract input types from parameters schema
        input_types = set()
        params = func.get('parameters', {})
        for prop_name, prop_def in params.get('properties', {}).items():
            input_types.add(prop_name)

        # Generate capability tags
        words = re.findall(r'[a-z]{3,}', description.lower())
        stop = {'the', 'and', 'for', 'with', 'from', 'that', 'this'}
        tags = {w for w in words if w not in stop}
        tags.add(name.lower().replace('_', ''))

        result.append(Tool(
            name=name,
            description=description,
            input_types=input_types,
            output_types=set(),
            capability_tags=tags,
        ))

    return result


def build_catalog_from_openai(functions: List[Dict]) -> ToolCatalog:
    """Build a ToolCatalog from OpenAI function definitions."""
    catalog = ToolCatalog()
    for tool in from_openai_functions(functions):
        catalog.register(tool)
    return catalog


def sc_route_openai(
    query: str,
    functions: List[Dict],
    catalog: Optional[ToolCatalog] = None,
) -> RoutingResult:
    """Route a query using SC-Router with OpenAI function definitions."""
    if catalog is None:
        catalog = build_catalog_from_openai(functions)
    return route(query, catalog)
