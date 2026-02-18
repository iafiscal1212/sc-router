"""Query decomposition for SC(1) routing.

Splits queries into executable sub-tasks with tool assignments.
Three modes:
  - Sequential: temporal markers ("first X, then Y, finally Z")
  - Parallel: conjunctions ("X and also Y")
  - Hierarchical: main task + sub-tasks
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .catalog import Tool, ToolCatalog
from .features import _extract_keywords, _split_subtasks


@dataclass
class SubTask:
    """A decomposed sub-task with tool assignment."""
    text: str
    tools: List[str] = field(default_factory=list)
    depends_on: List[int] = field(default_factory=list)  # indices of prerequisite subtasks
    mode: str = 'sequential'  # 'sequential', 'parallel', 'hierarchical'


@dataclass
class DecompositionResult:
    """Result of query decomposition."""
    subtasks: List[SubTask]
    mode: str  # overall decomposition mode
    original_query: str


# Temporal/sequential markers
_TEMPORAL = re.compile(
    r'\b(?:then|after that|next|finally|afterwards|subsequently|'
    r'luego|después de eso|siguiente|finalmente|posteriormente)\b',
    re.IGNORECASE,
)
_ORDINAL = re.compile(
    r'\b(?:first|second|third|fourth|fifth|'
    r'1st|2nd|3rd|4th|5th|'
    r'step\s*[1-5]|'
    r'primero|segundo|tercero|cuarto|quinto|'
    r'paso\s*[1-5])\b',
    re.IGNORECASE,
)

# Parallel markers
_PARALLEL = re.compile(
    r'\b(?:and also|at the same time|simultaneously|in parallel|'
    r'meanwhile|concurrently|'
    r'y también|al mismo tiempo|simultáneamente|en paralelo|'
    r'mientras tanto|concurrentemente)\b',
    re.IGNORECASE,
)

# List patterns
_NUMBERED_LIST = re.compile(r'(?:^|\n)\s*(\d+)[.)]\s*(.+?)(?=\n\s*\d+[.)]|\Z)', re.DOTALL)
_BULLET_LIST = re.compile(r'(?:^|\n)\s*[-*]\s*(.+?)(?=\n\s*[-*]|\Z)', re.DOTALL)


def decompose(query: str, catalog: ToolCatalog) -> DecompositionResult:
    """Decompose a query into sub-tasks with tool assignments.

    Tries decomposition modes in order:
    1. Numbered/bulleted list
    2. Sequential (temporal markers)
    3. Parallel (conjunction markers)
    4. Comma-separated
    5. Fallback: sentence splitting
    """
    # Try numbered list
    numbered = _NUMBERED_LIST.findall(query)
    if len(numbered) >= 2:
        return _build_sequential(
            [text.strip() for _, text in numbered],
            query, catalog,
        )

    # Try bullet list
    bullets = _BULLET_LIST.findall(query)
    if len(bullets) >= 2:
        return _build_sequential(
            [text.strip() for text in bullets],
            query, catalog,
        )

    # Check for temporal/sequential markers
    has_temporal = bool(_TEMPORAL.search(query))
    has_ordinal = bool(_ORDINAL.search(query))
    has_parallel = bool(_PARALLEL.search(query))

    if has_temporal or has_ordinal:
        parts = _split_by_temporal(query)
        if len(parts) >= 2:
            return _build_sequential(parts, query, catalog)

    if has_parallel:
        parts = _split_by_parallel(query)
        if len(parts) >= 2:
            return _build_parallel(parts, query, catalog)

    # Comma-separated items
    if ',' in query:
        parts = re.split(r',\s*(?:and\s+|y\s+)?', query)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]
        if len(parts) >= 2:
            return _build_parallel(parts, query, catalog)

    # Fallback: generic subtask splitting
    subtasks = _split_subtasks(query)
    if len(subtasks) >= 2:
        return _build_sequential(subtasks, query, catalog)

    # Single task — no decomposition needed
    tools = _match_tools(query, catalog)
    return DecompositionResult(
        subtasks=[SubTask(text=query, tools=tools, mode='sequential')],
        mode='single',
        original_query=query,
    )


def _split_by_temporal(query: str) -> List[str]:
    """Split query at temporal markers."""
    # Split at temporal words, keeping content
    parts = _TEMPORAL.split(query)
    # Also try ordinal splitting
    if len(parts) < 2:
        parts = _ORDINAL.split(query)
    parts = [p.strip().rstrip('.,;') for p in parts if p.strip() and len(p.strip()) > 3]
    return parts


def _split_by_parallel(query: str) -> List[str]:
    """Split query at parallel markers."""
    parts = _PARALLEL.split(query)
    parts = [p.strip().rstrip('.,;') for p in parts if p.strip() and len(p.strip()) > 3]
    return parts


def _match_tools(text: str, catalog: ToolCatalog) -> List[str]:
    """Match text to tool names."""
    keywords = _extract_keywords(text)
    tools = catalog.find_tools(keywords)
    return [t.name for t in tools]


def _build_sequential(
    parts: List[str],
    original: str,
    catalog: ToolCatalog,
) -> DecompositionResult:
    """Build a sequential decomposition where each subtask depends on the previous."""
    subtasks = []
    for i, text in enumerate(parts):
        tools = _match_tools(text, catalog)
        depends = [i - 1] if i > 0 else []
        subtasks.append(SubTask(
            text=text,
            tools=tools,
            depends_on=depends,
            mode='sequential',
        ))
    return DecompositionResult(
        subtasks=subtasks,
        mode='sequential',
        original_query=original,
    )


def _build_parallel(
    parts: List[str],
    original: str,
    catalog: ToolCatalog,
) -> DecompositionResult:
    """Build a parallel decomposition where subtasks are independent."""
    subtasks = []
    for text in parts:
        tools = _match_tools(text, catalog)
        subtasks.append(SubTask(
            text=text,
            tools=tools,
            depends_on=[],
            mode='parallel',
        ))
    return DecompositionResult(
        subtasks=subtasks,
        mode='parallel',
        original_query=original,
    )
