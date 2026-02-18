"""Bridge between SC-Router and selector-complexity.

Translates a query + tool catalog into a polynomial axiom system,
then uses the real SCPredictor and extract_features from
selector-complexity for mathematically grounded classification.

Optional: only active when selector-complexity is installed.
    pip install selector-complexity
"""

from typing import Dict, List, Optional, Tuple

from .catalog import Tool, ToolCatalog
from .features import _split_subtasks, _extract_keywords

# Cached selector-complexity predictor (fitted once on first use)
_sc_predictor = None


def _get_sc_predictor():
    """Get or create the cached selector-complexity SCPredictor."""
    global _sc_predictor
    if _sc_predictor is None:
        from selector_complexity import SCPredictor
        _sc_predictor = SCPredictor()
        _sc_predictor.fit_from_landscape()
    return _sc_predictor


def _catalog_to_axioms(
    query: str,
    catalog: ToolCatalog,
) -> Tuple[list, int]:
    """Translate a query + tool catalog into a polynomial axiom system.

    Encoding:
    - Each candidate tool is a variable (x_i).
    - Each sub-task generates a "pigeon" axiom: at least one tool must handle it.
      Encoded as: sum(x_i for matching tools) - 1 = 0
    - Incompatible tool pairs generate exclusion axioms:
      x_i * x_j = 0 (if tools i,j can't compose and share no tags)
    - Constraint keywords generate budget axioms:
      cost_i * x_i <= budget  →  sum(cost_i * x_i) - budget = 0

    Returns (axioms, num_vars) in selector-complexity format:
        axioms: list of list of (coef, frozenset) tuples
        num_vars: int
    """
    # Get candidate tools
    all_keywords = _extract_keywords(query)
    candidates = catalog.find_tools(all_keywords)

    if not candidates:
        candidates = catalog.tools[:5]  # fallback: first 5 tools

    num_vars = len(candidates)
    if num_vars == 0:
        return [[(1.0, frozenset())]], 1  # trivial system

    tool_idx = {t.name: i for i, t in enumerate(candidates)}
    axioms = []

    # Sub-task coverage axioms: each sub-task needs at least one tool
    subtasks = _split_subtasks(query)
    for st in subtasks:
        st_keywords = _extract_keywords(st)
        matched = catalog.find_tools(st_keywords)
        matched = [t for t in matched if t.name in tool_idx]

        if matched:
            # sum(x_i for matching tools) - 1 = 0
            terms = [(1.0, frozenset({tool_idx[t.name]})) for t in matched]
            terms.append((-1.0, frozenset()))  # constant -1
            axioms.append(terms)

    # Compatibility axioms: incompatible pairs get exclusion
    graph = catalog.compatibility_graph()
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            # If no shared tags and can't compose either way → exclusion
            if (b.name not in graph.get(a.name, {})
                    and not catalog.can_compose(a.name, b.name)
                    and not catalog.can_compose(b.name, a.name)):
                # x_i * x_j = 0
                axioms.append([(1.0, frozenset({i, j}))])

    # If no axioms generated, add a trivial one
    if not axioms:
        axioms.append([(1.0, frozenset({0})), (-1.0, frozenset())])

    return axioms, num_vars


def classify_with_sc(
    query: str,
    catalog: ToolCatalog,
) -> Optional[Dict]:
    """Classify a query using the real selector-complexity framework.

    Translates the query + catalog into a polynomial system, then uses
    selector-complexity's extract_features + SCPredictor for classification.

    Returns None if selector-complexity is not installed.
    Returns dict with 'level', 'confidence', 'features', 'method'.
    """
    try:
        from selector_complexity import extract_features, SCPredictor
    except ImportError:
        return None

    axioms, num_vars = _catalog_to_axioms(query, catalog)

    # Extract the 17 real features from the polynomial system
    sc_features = extract_features(axioms, num_vars)

    # Use the real predictor (cached at module level)
    predictor = _get_sc_predictor()
    prediction = predictor.predict(sc_features)

    return {
        'level': prediction['predicted_level'],
        'confidence': prediction.get('confidence', 0.0),
        'features': sc_features,
        'method': 'selector_complexity_bridge',
        'axiom_system_size': {
            'num_axioms': len(axioms),
            'num_vars': num_vars,
        },
    }


def hardness_score(query: str, catalog: ToolCatalog) -> Optional[Dict]:
    """Compute hardness score using selector-complexity's quantify_hardness.

    Returns None if selector-complexity is not installed.
    """
    try:
        from selector_complexity import quantify_hardness
    except ImportError:
        return None

    axioms, num_vars = _catalog_to_axioms(query, catalog)
    return quantify_hardness(axioms, num_vars, max_degree=4)


def is_available() -> bool:
    """Check if selector-complexity is installed."""
    try:
        import selector_complexity
        return True
    except ImportError:
        return False
