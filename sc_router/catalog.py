"""Tool catalog and compatibility graph.

Tools are variables, capability overlaps create the constraint graph.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Tool:
    """A tool available for routing.

    Attributes:
        name: Unique identifier.
        description: What the tool does (used for keyword matching).
        input_types: Types this tool accepts.
        output_types: Types this tool produces.
        capability_tags: Keywords describing capabilities.
        cost: Relative cost of invoking this tool (default 1.0).
    """
    name: str
    description: str
    input_types: Set[str] = field(default_factory=set)
    output_types: Set[str] = field(default_factory=set)
    capability_tags: Set[str] = field(default_factory=set)
    cost: float = 1.0


class ToolCatalog:
    """Registry of tools with compatibility graph construction.

    Tools are variables, shared capability tags create edges in the
    constraint graph.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._graph_cache: Optional[dict] = None

    def register(self, tool: Tool) -> None:
        """Register a tool in the catalog."""
        self._tools[tool.name] = tool
        self._graph_cache = None  # invalidate cache

    def unregister(self, name: str) -> None:
        """Remove a tool from the catalog."""
        self._tools.pop(name, None)
        self._graph_cache = None

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    @property
    def tools(self) -> List[Tool]:
        """All registered tools."""
        return list(self._tools.values())

    @property
    def size(self) -> int:
        return len(self._tools)

    def find_tools(self, keywords: Set[str]) -> List[Tool]:
        """Find tools whose capability_tags or description match any keyword.

        Uses case-insensitive matching against tags and description words.
        """
        keywords_lower = {k.lower() for k in keywords}
        results = []
        for tool in self._tools.values():
            tags_lower = {t.lower() for t in tool.capability_tags}
            if tags_lower & keywords_lower:
                results.append(tool)
                continue
            desc_words = {w.lower().strip(".,;:!?()") for w in tool.description.split()}
            if desc_words & keywords_lower:
                results.append(tool)
        return results

    def compatibility_graph(self) -> Dict[str, Dict[str, Set[str]]]:
        """Build the compatibility graph.

        Nodes = tool names. An edge exists between tools A and B if:
        - They share capability tags, OR
        - A's output_types overlap with B's input_types (composability).

        Returns dict: {tool_name: {neighbor_name: shared_tags_or_types}}.
        """
        if self._graph_cache is not None:
            return self._graph_cache

        names = list(self._tools.keys())
        graph: Dict[str, Dict[str, Set[str]]] = {n: {} for n in names}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = self._tools[names[i]]
                b = self._tools[names[j]]
                shared = set()
                # Shared capability tags
                tag_overlap = a.capability_tags & b.capability_tags
                shared.update(tag_overlap)
                # Composability: A->B or B->A
                if a.output_types & b.input_types:
                    shared.add(f"compose:{a.name}->{b.name}")
                if b.output_types & a.input_types:
                    shared.add(f"compose:{b.name}->{a.name}")
                if shared:
                    graph[a.name][b.name] = shared
                    graph[b.name][a.name] = shared

        self._graph_cache = graph
        return graph

    def can_compose(self, a_name: str, b_name: str) -> bool:
        """Check if tool A's output can feed into tool B's input."""
        a = self._tools.get(a_name)
        b = self._tools.get(b_name)
        if not a or not b:
            return False
        return bool(a.output_types & b.input_types)

    def composable_chains(self, tool_names: List[str]) -> List[List[str]]:
        """Find valid sequential chains among the given tools."""
        if len(tool_names) <= 1:
            return [tool_names] if tool_names else []

        chains = []
        # Try all permutations for small sets, greedy for larger
        if len(tool_names) <= 6:
            from itertools import permutations
            for perm in permutations(tool_names):
                valid = True
                for k in range(len(perm) - 1):
                    if not self.can_compose(perm[k], perm[k + 1]):
                        valid = False
                        break
                if valid:
                    chains.append(list(perm))
        return chains
