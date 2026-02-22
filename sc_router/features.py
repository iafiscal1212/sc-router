"""Extract 17 structural features from a query + tool catalog.

Analyzes query text + tool compatibility graph.
All computed with text parsing (regex, keywords, sentence splitting) + graph properties.
No ML dependencies.
"""

import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .catalog import Tool, ToolCatalog


# --- Text parsing utilities ---

# Sentence/sub-task splitters
_CONJUNCTIONS = re.compile(
    r'\b(?:and|then|after that|next|also|additionally|moreover|furthermore|'
    r'y|luego|después|además|también)\b',
    re.IGNORECASE,
)
_SEQUENTIAL_MARKERS = re.compile(
    r'\b(?:first|second|third|then|next|finally|step\s*\d+|'
    r'primero|segundo|tercero|luego|después|finalmente|paso\s*\d+)\b',
    re.IGNORECASE,
)
_LIST_PATTERN = re.compile(r'(?:^|\n)\s*(?:\d+[.)]\s*|-\s*|\*\s*)')
_COMMA_LIST = re.compile(r',\s*(?:and\s+|y\s+)?')

# Constraint/requirement indicators
_CONSTRAINT_WORDS = re.compile(
    r'\b(?:must|should|need|require|only|within|under|below|above|between|'
    r'at least|at most|no more than|budget|limit|maximum|minimum|'
    r'debe|necesita|requiere|solo|dentro|bajo|sobre|entre|'
    r'al menos|como máximo|presupuesto|límite|máximo|mínimo)\b',
    re.IGNORECASE,
)


def _split_subtasks(query: str) -> List[str]:
    """Split a query into sub-tasks using multiple heuristics."""
    subtasks = []

    # Try numbered/bulleted list first
    list_items = _LIST_PATTERN.split(query)
    list_items = [s.strip() for s in list_items if s.strip()]
    if len(list_items) > 1:
        return list_items

    # Try sentence splitting (., ;, newline)
    sentences = re.split(r'[.;!\n]+', query)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

    if len(sentences) > 1:
        # Further split long sentences by conjunctions
        expanded = []
        for s in sentences:
            parts = _CONJUNCTIONS.split(s)
            parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
            expanded.extend(parts)
        return expanded if len(expanded) > 1 else sentences

    # Single sentence: try conjunction splitting
    parts = _CONJUNCTIONS.split(query)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
    if len(parts) > 1:
        return parts

    # Try comma-separated items
    if ',' in query:
        parts = _COMMA_LIST.split(query)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
        if len(parts) > 1:
            return parts

    return [query.strip()]


def _extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text, filtering stop words.

    Applies bilingual synonym expansion (ES→EN) so that Spanish queries
    can match tools with English capability tags.
    """
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'shall', 'can',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me',
        'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they',
        'what', 'which', 'who', 'how', 'when', 'where', 'why',
        'not', 'no', 'nor', 'but', 'or', 'so', 'if', 'up',
        'el', 'la', 'los', 'las', 'un', 'una', 'de', 'en', 'con',
        'por', 'para', 'es', 'son', 'fue', 'ser', 'estar', 'que',
        'hace', 'hay', 'qué', 'cómo', 'cuánto',
    }
    words = re.findall(r'[a-záéíóúñ]+', text.lower())
    keywords = {w for w in words if w not in stop_words and len(w) > 2}
    # Expand Spanish synonyms to English equivalents
    expanded = set()
    for kw in keywords:
        expanded.add(kw)
        if kw in _ES_EN_SYNONYMS:
            expanded.update(_ES_EN_SYNONYMS[kw])
    return expanded


# Common Spanish→English keyword mappings for tool matching
_ES_EN_SYNONYMS: Dict[str, Set[str]] = {
    'tiempo': {'weather', 'forecast'},
    'clima': {'weather', 'climate', 'forecast'},
    'temperatura': {'temperature', 'weather'},
    'buscar': {'search', 'find', 'lookup'},
    'búsqueda': {'search', 'find'},
    'calcular': {'calculate', 'math', 'arithmetic'},
    'cálculo': {'calculate', 'math'},
    'traducir': {'translate', 'translation'},
    'traducción': {'translate', 'translation'},
    'resumir': {'summarize', 'summary'},
    'resumen': {'summarize', 'summary'},
    'vuelos': {'flight', 'flights', 'travel'},
    'vuelo': {'flight', 'flights', 'travel'},
    'hotel': {'hotel', 'hotels', 'accommodation'},
    'hoteles': {'hotel', 'hotels', 'accommodation'},
    'restaurante': {'restaurant', 'restaurants', 'dining'},
    'restaurantes': {'restaurant', 'restaurants', 'dining'},
    'presupuesto': {'budget', 'cost'},
    'viaje': {'travel', 'trip'},
    'viajar': {'travel', 'trip'},
    'sentimiento': {'sentiment', 'opinion'},
    'analizar': {'analyze', 'analysis'},
    'análisis': {'analyze', 'analysis'},
    'tendencia': {'trend', 'trends'},
    'tendencias': {'trend', 'trends'},
    'idioma': {'language', 'translate'},
    'comida': {'food', 'restaurant', 'dining'},
    'precio': {'price', 'cost'},
    'precios': {'price', 'cost'},
    'reservar': {'book', 'reserve'},
    'reserva': {'book', 'reserve'},
    'optimizar': {'optimize', 'budget'},
    'noticias': {'news', 'information'},
    'información': {'information', 'search'},
    'mercado': {'market', 'trend', 'trends'},
    'datos': {'data', 'information'},
    'modelo': {'model', 'predict', 'forecast'},
    'predictivo': {'predict', 'forecast', 'model'},
    'correlacionar': {'correlate', 'analyze', 'cross-reference'},
    'cruzar': {'cross-reference', 'correlate', 'combine'},
    'construir': {'build', 'create'},
    'redes': {'social', 'web'},
    'sociales': {'social', 'sentiment'},
}


def _count_constraints(query: str) -> int:
    """Count constraint/requirement phrases in the query."""
    return len(_CONSTRAINT_WORDS.findall(query))


def _match_subtask_to_tools(
    subtask: str,
    catalog: ToolCatalog,
) -> List[Tool]:
    """Find tools matching a sub-task's keywords."""
    keywords = _extract_keywords(subtask)
    return catalog.find_tools(keywords)


def _bfs_components(graph: Dict[str, Dict]) -> List[Set[str]]:
    """Find connected components via BFS."""
    visited = set()
    components = []
    for node in graph:
        if node in visited:
            continue
        component = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in graph.get(current, {}):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)
    return components


def _spectral_gap(graph: Dict[str, Dict], nodes: List[str]) -> float:
    """Compute normalized Laplacian spectral gap (requires numpy, optional)."""
    n = len(nodes)
    if n <= 1:
        return 0.0
    try:
        import numpy as np
    except ImportError:
        return 0.0

    idx = {name: i for i, name in enumerate(nodes)}
    adj = np.zeros((n, n))
    for u in nodes:
        for v in graph.get(u, {}):
            if v in idx:
                adj[idx[u]][idx[v]] = 1.0

    degrees = adj.sum(axis=1)
    laplacian = np.diag(degrees) - adj
    # Normalized Laplacian
    for i in range(n):
        if degrees[i] > 0:
            for j in range(n):
                if i == j:
                    laplacian[i][j] = 1.0 if degrees[i] > 0 else 0.0
                elif adj[i][j] > 0:
                    laplacian[i][j] = -1.0 / (degrees[i] * degrees[j]) ** 0.5
                else:
                    laplacian[i][j] = 0.0

    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues.sort()
    # Second smallest eigenvalue
    if len(eigenvalues) >= 2:
        return float(eigenvalues[1])
    return 0.0


def extract_query_features(query: str, catalog: ToolCatalog) -> Dict[str, float]:
    """Extract 17 structural features from a query + tool catalog.

    Returns dict with all 17 feature values.
    """
    # --- Step 1: Parse query into sub-tasks ---
    subtasks = _split_subtasks(query)
    num_subtasks = len(subtasks)

    # --- Step 2: Match sub-tasks to tools ---
    all_keywords = _extract_keywords(query)
    candidate_tools = catalog.find_tools(all_keywords)
    candidate_names = {t.name for t in candidate_tools}

    subtask_tools: List[List[Tool]] = []
    for st in subtasks:
        matched = _match_subtask_to_tools(st, catalog)
        # Filter to only candidates relevant to full query
        matched = [t for t in matched if t.name in candidate_names] or matched
        subtask_tools.append(matched)

    # --- Step 3: Build subgraph of candidate tools ---
    full_graph = catalog.compatibility_graph()
    # Subgraph restricted to candidate tools
    subgraph: Dict[str, Dict] = {}
    for name in candidate_names:
        subgraph[name] = {}
        for neighbor, shared in full_graph.get(name, {}).items():
            if neighbor in candidate_names:
                subgraph[name][neighbor] = shared

    # --- Step 4: Compute 17 features ---
    num_candidates = max(len(candidate_tools), 1)

    # F1: num_constraints — requirements/constraints in the query
    num_constraints = _count_constraints(query)

    # F2: num_candidate_tools
    num_candidate_tools = len(candidate_tools)

    # F3: max_dependency_depth — longest chain of composable tools
    max_dep_depth = 0
    for st_tools in subtask_tools:
        depth = 0
        names_list = [t.name for t in st_tools]
        for i in range(len(names_list)):
            chain_len = 1
            current = names_list[i]
            visited_chain = {current}
            for j in range(len(names_list)):
                if names_list[j] not in visited_chain and catalog.can_compose(current, names_list[j]):
                    chain_len += 1
                    current = names_list[j]
                    visited_chain.add(current)
            depth = max(depth, chain_len)
        max_dep_depth = max(max_dep_depth, depth)

    # F4: avg_dependency_depth
    depths = []
    for st_tools in subtask_tools:
        names_list = [t.name for t in st_tools]
        best = 1
        for i in range(len(names_list)):
            chain_len = 1
            current = names_list[i]
            visited_chain = {current}
            for j in range(len(names_list)):
                if names_list[j] not in visited_chain and catalog.can_compose(current, names_list[j]):
                    chain_len += 1
                    current = names_list[j]
                    visited_chain.add(current)
            best = max(best, chain_len)
        depths.append(best)
    avg_dep_depth = sum(depths) / max(len(depths), 1)

    # F5: tool_coverage — fraction of catalog that is relevant
    tool_coverage = len(candidate_tools) / max(catalog.size, 1)

    # F6: tool_overlap — fraction of candidate pairs sharing tags
    num_pairs = num_candidates * (num_candidates - 1) / 2 if num_candidates > 1 else 1
    overlap_count = 0
    cand_list = list(candidate_tools)
    for i in range(len(cand_list)):
        for j in range(i + 1, len(cand_list)):
            if cand_list[i].capability_tags & cand_list[j].capability_tags:
                overlap_count += 1
    tool_overlap = overlap_count / num_pairs if num_pairs > 0 else 0.0

    # F7: query_expansion — ratio of implicit sub-tasks to explicit markers
    explicit_markers = len(_SEQUENTIAL_MARKERS.findall(query))
    query_expansion = num_subtasks / max(explicit_markers, 1)

    # F8: constraint_tool_ratio — constraints per candidate tool
    constraint_tool_ratio = num_constraints / max(num_candidates, 1)

    # F9-F11: task breadth distribution
    single_tool_tasks = sum(1 for st in subtask_tools if len(st) == 1)
    two_tool_tasks = sum(1 for st in subtask_tools if len(st) == 2)
    multi_tool_tasks = sum(1 for st in subtask_tools if len(st) >= 3)

    # F12: avg_task_breadth — average tools per sub-task
    breadths = [len(st) for st in subtask_tools]
    avg_task_breadth = sum(breadths) / max(len(breadths), 1)

    # F13: max_task_breadth
    max_task_breadth = max(breadths) if breadths else 0

    # F14: tool_clusters — connected components of the candidate subgraph
    if subgraph:
        components = _bfs_components(subgraph)
    else:
        components = []
    tool_clusters = len(components)

    # F15: max_cluster_size
    max_cluster_size = max((len(c) for c in components), default=0)

    # F16: spectral_gap (only for catalogs <= 200 candidates)
    if num_candidates <= 200 and subgraph:
        spectral_gap = _spectral_gap(subgraph, list(candidate_names))
    else:
        spectral_gap = 0.0

    # F17: avg_tool_specificity — how specialized the candidate tools are
    # Higher = more specific (fewer tags per tool, more unique tags)
    specificities = []
    all_tags = set()
    for t in candidate_tools:
        all_tags.update(t.capability_tags)
    total_tags = max(len(all_tags), 1)
    for t in candidate_tools:
        if t.capability_tags:
            # Specificity = 1 - (fraction of total unique tags this tool covers)
            specificities.append(1.0 - len(t.capability_tags) / total_tags)
        else:
            specificities.append(0.5)
    avg_tool_specificity = sum(specificities) / max(len(specificities), 1)

    return {
        'num_constraints': float(num_constraints),
        'num_candidate_tools': float(num_candidate_tools),
        'max_dependency_depth': float(max_dep_depth),
        'avg_dependency_depth': avg_dep_depth,
        'tool_coverage': tool_coverage,
        'tool_overlap': tool_overlap,
        'query_expansion': query_expansion,
        'constraint_tool_ratio': constraint_tool_ratio,
        'single_tool_tasks': float(single_tool_tasks),
        'two_tool_tasks': float(two_tool_tasks),
        'multi_tool_tasks': float(multi_tool_tasks),
        'avg_task_breadth': avg_task_breadth,
        'max_task_breadth': float(max_task_breadth),
        'tool_clusters': float(tool_clusters),
        'max_cluster_size': float(max_cluster_size),
        'spectral_gap': spectral_gap,
        'avg_tool_specificity': avg_tool_specificity,
    }


# Ordered list of feature names (for vector operations)
FEATURE_NAMES = [
    'num_constraints', 'num_candidate_tools', 'max_dependency_depth',
    'avg_dependency_depth', 'tool_coverage', 'tool_overlap',
    'query_expansion', 'constraint_tool_ratio', 'single_tool_tasks',
    'two_tool_tasks', 'multi_tool_tasks', 'avg_task_breadth',
    'max_task_breadth', 'tool_clusters', 'max_cluster_size',
    'spectral_gap', 'avg_tool_specificity',
]
