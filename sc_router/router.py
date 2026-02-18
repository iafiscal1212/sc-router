"""Main routing engine.

Analogous to strategy.py in selector-complexity:
uses SC classification to choose the routing strategy, then executes it.

Strategies:
  SC(0) → direct dispatch (single tool)
  SC(1) → decompose and route (pipeline)
  SC(2) → search combinations (constraint satisfaction)
  SC(3) → delegate to agent (full autonomy)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .catalog import Tool, ToolCatalog
from .classifier import classify_query
from .decomposer import decompose
from .features import _extract_keywords


@dataclass
class ToolAssignment:
    """A tool assigned to a sub-task."""
    tool: str
    subtask: str
    confidence: str = 'medium'
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Complete routing result."""
    sc_level: int
    strategy: str
    tool_assignments: List[ToolAssignment]
    classification: Dict = field(default_factory=dict)
    decomposition: Optional[Any] = None
    metadata: Dict = field(default_factory=dict)


def route(
    query: str,
    catalog: ToolCatalog,
    agent_callback: Optional[Callable] = None,
    predictor=None,
) -> RoutingResult:
    """Route a query to the appropriate tools based on SC classification.

    Args:
        query: The user query to route.
        catalog: Tool catalog with registered tools.
        agent_callback: Optional callback for SC(3) agent delegation.
            Signature: callback(query, catalog, classification) -> RoutingResult
        predictor: Optional custom SCRouterPredictor instance.

    Returns:
        RoutingResult with strategy and tool assignments.
    """
    classification = classify_query(query, catalog, predictor=predictor)
    level = classification['level']

    if level == 0:
        return _direct_dispatch(query, catalog, classification)
    elif level == 1:
        return _decompose_and_route(query, catalog, classification)
    elif level == 2:
        return _search_combinations(query, catalog, classification)
    else:  # level == 3
        return _delegate_to_agent(query, catalog, classification, agent_callback)


def _direct_dispatch(
    query: str,
    catalog: ToolCatalog,
    classification: Dict,
) -> RoutingResult:
    """SC(0): Direct dispatch to single best tool."""
    keywords = _extract_keywords(query)
    candidates = catalog.find_tools(keywords)

    if not candidates:
        # No tool found — still return result
        return RoutingResult(
            sc_level=0,
            strategy='direct',
            tool_assignments=[],
            classification=classification,
            metadata={'note': 'no matching tool found'},
        )

    # Pick best candidate (most tag overlap)
    best = candidates[0]
    best_overlap = 0
    for tool in candidates:
        overlap = len(tool.capability_tags & keywords)
        if overlap > best_overlap:
            best_overlap = overlap
            best = tool

    return RoutingResult(
        sc_level=0,
        strategy='direct',
        tool_assignments=[ToolAssignment(
            tool=best.name,
            subtask=query,
            confidence='high' if len(candidates) == 1 else 'medium',
        )],
        classification=classification,
    )


def _decompose_and_route(
    query: str,
    catalog: ToolCatalog,
    classification: Dict,
) -> RoutingResult:
    """SC(1): Decompose query into sub-tasks, route each."""
    decomposition = decompose(query, catalog)
    assignments = []

    for st in decomposition.subtasks:
        if st.tools:
            # Pick the best tool for this subtask
            best_tool = st.tools[0]
            assignments.append(ToolAssignment(
                tool=best_tool,
                subtask=st.text,
                confidence='high' if len(st.tools) == 1 else 'medium',
            ))
        else:
            assignments.append(ToolAssignment(
                tool='',
                subtask=st.text,
                confidence='low',
                parameters={'note': 'no matching tool'},
            ))

    return RoutingResult(
        sc_level=1,
        strategy=f'pipeline_{decomposition.mode}',
        tool_assignments=assignments,
        classification=classification,
        decomposition=decomposition,
    )


def _search_combinations(
    query: str,
    catalog: ToolCatalog,
    classification: Dict,
) -> RoutingResult:
    """SC(2): Search tool combinations under constraints.

    Explores composable tool chains, scoring by:
    - Keyword coverage of the original query
    - Composability (output→input chaining)
    - Cost minimization
    """
    keywords = _extract_keywords(query)
    candidates = catalog.find_tools(keywords)

    if not candidates:
        return RoutingResult(
            sc_level=2,
            strategy='search',
            tool_assignments=[],
            classification=classification,
            metadata={'note': 'no candidates found'},
        )

    # Decompose to understand structure
    decomposition = decompose(query, catalog)

    # Score candidate combinations
    # For each subtask, rank tools by relevance
    assignments = []
    used_tools = set()

    for st in decomposition.subtasks:
        st_keywords = _extract_keywords(st.text)
        scored = []
        for tool in candidates:
            tag_overlap = len(tool.capability_tags & st_keywords)
            desc_overlap = len({w.lower().strip('.,;:') for w in tool.description.split()} & st_keywords)
            # Prefer unused tools (diversity)
            diversity_bonus = 0.5 if tool.name not in used_tools else 0.0
            score = tag_overlap + desc_overlap * 0.5 + diversity_bonus
            scored.append((tool, score))

        scored.sort(key=lambda x: -x[1])

        if scored and scored[0][1] > 0:
            best = scored[0][0]
            used_tools.add(best.name)
            assignments.append(ToolAssignment(
                tool=best.name,
                subtask=st.text,
                confidence='medium',
            ))
        else:
            assignments.append(ToolAssignment(
                tool='',
                subtask=st.text,
                confidence='low',
                parameters={'note': 'no clear match'},
            ))

    # Check composability of the assignment chain
    composable_chain = True
    for i in range(len(assignments) - 1):
        if assignments[i].tool and assignments[i + 1].tool:
            if not catalog.can_compose(assignments[i].tool, assignments[i + 1].tool):
                composable_chain = False
                break

    return RoutingResult(
        sc_level=2,
        strategy='search',
        tool_assignments=assignments,
        classification=classification,
        decomposition=decomposition,
        metadata={
            'composable_chain': composable_chain,
            'candidates_explored': len(candidates),
        },
    )


def _delegate_to_agent(
    query: str,
    catalog: ToolCatalog,
    classification: Dict,
    agent_callback: Optional[Callable],
) -> RoutingResult:
    """SC(3): Delegate to external agent for complex routing.

    If agent_callback is provided, delegates to it.
    Otherwise, returns a routing result indicating agent needed.
    """
    if agent_callback:
        return agent_callback(query, catalog, classification)

    # No agent — return all candidate tools as potential assignments
    keywords = _extract_keywords(query)
    candidates = catalog.find_tools(keywords)

    assignments = [
        ToolAssignment(
            tool=t.name,
            subtask=query,
            confidence='low',
            parameters={'role': 'candidate'},
        )
        for t in candidates
    ]

    return RoutingResult(
        sc_level=3,
        strategy='agent_required',
        tool_assignments=assignments,
        classification=classification,
        metadata={
            'note': 'Query requires full agent orchestration. '
                    'Provide agent_callback for autonomous execution.',
            'candidates': [t.name for t in candidates],
        },
    )
