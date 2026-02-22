"""Pattern detection shortcuts for fast SC classification.

Detects structural patterns in queries that allow immediate SC classification
without running the full feature extraction + prediction pipeline.
"""

import re
from typing import Dict, List, Optional

from .catalog import ToolCatalog


# --- Pattern regexes ---

_SINGLE_QUESTION = re.compile(
    r'^(?:what|how|when|where|who|which|is|are|does|do|can|will|'
    r'qué|cómo|cuándo|dónde|quién|cuál|es|son|puede)\b',
    re.IGNORECASE,
)

_SEQUENTIAL_WORDS = re.compile(
    r'\b(?:then|after|next|first|second|third|finally|step|'
    r'luego|después|siguiente|primero|segundo|tercero|finalmente|paso)\b',
    re.IGNORECASE,
)

_PARALLEL_WORDS = re.compile(
    r'\b(?:and also|at the same time|simultaneously|in parallel|both|'
    r'y también|al mismo tiempo|simultáneamente|en paralelo|ambos)\b',
    re.IGNORECASE,
)

_CONSTRAINT_HEAVY = re.compile(
    r'\b(?:budget|limit|within|under|between|at least|at most|no more|'
    r'optimize|best|cheapest|fastest|maximize|minimize|'
    r'presupuesto|límite|dentro|bajo|entre|al menos|como máximo|'
    r'optimizar|mejor|más barato|más rápido|maximizar|minimizar)\b',
    re.IGNORECASE,
)

_CROSS_REFERENCE = re.compile(
    r'\b(?:cross.?reference|correlate|combine with|integrate|'
    r'merge|synthesize|compare across|overlay|'
    r'cruzar|correlacionar|combinar con|integrar|'
    r'fusionar|sintetizar|comparar entre|superponer)\b',
    re.IGNORECASE,
)

_ANALYSIS_CHAIN = re.compile(
    r'\b(?:analyze|build model|predict|forecast|train|'
    r'analizar|construir modelo|predecir|pronosticar|entrenar)\b',
    re.IGNORECASE,
)


def detect_single_tool_pattern(query: str, catalog: ToolCatalog) -> Optional[Dict]:
    """Detect SC(0): single tool, obvious dispatch.

    Triggers when:
    - Query is a simple question/command
    - Only 1 tool matches keywords
    - No sequential or parallel markers
    """
    # Short query, single question form
    words = query.split()
    if len(words) > 20:
        return None

    if not _SINGLE_QUESTION.match(query.strip()):
        # Also accept imperative commands like "Get X", "Calculate Y"
        if len(words) > 10:
            return None

    # No composition markers
    if _SEQUENTIAL_WORDS.search(query) or _PARALLEL_WORDS.search(query):
        return None

    # Check tool matching
    from .features import _extract_keywords
    keywords = _extract_keywords(query)
    matched = catalog.find_tools(keywords)

    if len(matched) == 1:
        return {
            'pattern': 'single_tool',
            'level': 0,
            'confidence': 'high',
            'matched_tool': matched[0].name,
        }
    elif len(matched) == 0 and len(words) <= 8:
        return {
            'pattern': 'single_tool_no_match',
            'level': 0,
            'confidence': 'medium',
            'matched_tool': None,
        }

    return None


def detect_pipeline_pattern(query: str, catalog: ToolCatalog) -> Optional[Dict]:
    """Detect SC(1): sequential or parallel pipeline.

    Triggers when:
    - Clear sequential markers ("first X, then Y, finally Z")
    - Or parallel markers ("do X and also Y")
    - Each sub-task maps to 1-2 tools
    """
    seq_markers = _SEQUENTIAL_WORDS.findall(query)
    par_markers = _PARALLEL_WORDS.findall(query)

    if not seq_markers and not par_markers:
        return None

    # Split into sub-tasks
    from .features import _split_subtasks, _extract_keywords
    subtasks = _split_subtasks(query)

    if len(subtasks) < 2:
        return None

    # Check each sub-task maps to few tools
    subtask_matches = []
    too_ambiguous = 0
    for st in subtasks:
        kw = _extract_keywords(st)
        tools = catalog.find_tools(kw)
        if len(tools) > catalog.size // 2:
            return None  # Matches too many tools — not a pipeline
        subtask_matches.append([t.name for t in tools])

    mode = 'sequential' if len(seq_markers) >= len(par_markers) else 'parallel'

    return {
        'pattern': f'pipeline_{mode}',
        'level': 1,
        'confidence': 'high' if len(seq_markers) + len(par_markers) >= 2 else 'medium',
        'subtasks': len(subtasks),
        'subtask_tools': subtask_matches,
    }


def detect_topology_pattern(query: str, catalog: ToolCatalog) -> Optional[Dict]:
    """Detect SC level from query topology analysis.

    Analyzes the structure of the dependency graph:
    - Linear chain → SC(0-1)
    - Tree/DAG → SC(1-2)
    - Dense/cyclic → SC(2-3)
    """
    from .features import _extract_keywords, _split_subtasks, _bfs_components

    subtasks = _split_subtasks(query)
    if len(subtasks) <= 1:
        return None

    # Build dependency info
    all_keywords = _extract_keywords(query)
    candidates = catalog.find_tools(all_keywords)
    if len(candidates) < 2:
        return None

    graph = catalog.compatibility_graph()
    cand_names = {t.name for t in candidates}

    # Build subgraph
    subgraph = {}
    edge_count = 0
    for name in cand_names:
        subgraph[name] = {}
        for neighbor, shared in graph.get(name, {}).items():
            if neighbor in cand_names:
                subgraph[name][neighbor] = shared
                edge_count += 1
    edge_count //= 2  # undirected

    n = len(cand_names)
    if n < 2:
        return None

    max_edges = n * (n - 1) // 2
    density = edge_count / max_edges if max_edges > 0 else 0

    components = _bfs_components(subgraph)
    degrees = [len(subgraph.get(name, {})) for name in cand_names]
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0

    # Cross-reference detection (SC3 indicator)
    has_cross_ref = bool(_CROSS_REFERENCE.search(query))
    has_analysis = bool(_ANALYSIS_CHAIN.search(query))

    # Classification
    if has_cross_ref and has_analysis:
        return {
            'pattern': 'entangled',
            'level': 3,
            'confidence': 'high',
            'density': density,
            'components': len(components),
        }

    constraint_count = len(_CONSTRAINT_HEAVY.findall(query))

    if density > 0.7 and constraint_count >= 2:
        return {
            'pattern': 'dense_constrained',
            'level': 2,
            'confidence': 'medium',
            'density': density,
            'constraints': constraint_count,
        }

    if density < 0.3 and len(components) <= 2:
        level = 1 if len(subtasks) > 1 else 0
        return {
            'pattern': 'sparse_decomposable',
            'level': level,
            'confidence': 'medium',
            'density': density,
            'components': len(components),
        }

    if density > 0.5 or (has_cross_ref or has_analysis):
        return {
            'pattern': 'moderately_connected',
            'level': 2,
            'confidence': 'low',
            'density': density,
        }

    return None


def detect_query_patterns(query: str, catalog: ToolCatalog) -> Dict:
    """Run all pattern detectors with priority logic.



    Priority:
    1. Single tool (SC0) — fastest shortcut
    2. Pipeline (SC1) — clear sequential/parallel markers
    3. Topology (SC0-3) — graph structure analysis

    Returns dict with shortcut info or no-shortcut indication.
    """
    # Priority 1: Single tool
    single = detect_single_tool_pattern(query, catalog)
    if single and single.get('confidence') == 'high':
        return {
            'shortcut_available': True,
            'shortcut_level': single['level'],
            'shortcut_confidence': single['confidence'],
            'shortcut_source': single['pattern'],
            'details': single,
        }

    # Priority 2: Pipeline
    pipeline = detect_pipeline_pattern(query, catalog)
    if pipeline and pipeline.get('confidence') in ('high', 'medium'):
        return {
            'shortcut_available': True,
            'shortcut_level': pipeline['level'],
            'shortcut_confidence': pipeline['confidence'],
            'shortcut_source': pipeline['pattern'],
            'details': pipeline,
        }

    # Priority 3: Topology
    topology = detect_topology_pattern(query, catalog)
    if topology and topology.get('confidence') in ('high', 'medium'):
        return {
            'shortcut_available': True,
            'shortcut_level': topology['level'],
            'shortcut_confidence': topology['confidence'],
            'shortcut_source': topology['pattern'],
            'details': topology,
        }

    # No shortcut — need full analysis
    return {
        'shortcut_available': False,
        'shortcut_level': None,
        'shortcut_confidence': None,
        'shortcut_source': None,
        'details': {
            'single_tool': single,
            'pipeline': pipeline,
            'topology': topology,
        },
    }
