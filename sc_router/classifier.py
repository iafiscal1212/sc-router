"""4-phase SC classification pipeline for queries.

Analogous to classifier.py in selector-complexity:
  Phase 1: Structural analysis (extract_query_features)
  Phase 2: Pattern shortcuts (detect_query_patterns)
  Phase 3: Tool matching analysis
  Phase 4: SC(0-3) classification

Fast-exit shortcuts at Phase 2 and Phase 3 avoid unnecessary computation.
"""

from typing import Dict, Optional

from .catalog import ToolCatalog
from .features import extract_query_features, _extract_keywords, _split_subtasks
from .patterns import detect_query_patterns
from .predictor import SCRouterPredictor

# Module-level cached predictor (fitted on first use)
_predictor: Optional[SCRouterPredictor] = None


def _get_predictor() -> SCRouterPredictor:
    """Get or create the module-level predictor."""
    global _predictor
    if _predictor is None:
        _predictor = SCRouterPredictor().fit_from_examples()
    return _predictor


def classify_query(
    query: str,
    catalog: ToolCatalog,
    predictor: Optional[SCRouterPredictor] = None,
) -> Dict:
    """Classify a query's routing complexity as SC(0-3).

    4-phase pipeline with fast-exit shortcuts:

    Phase 1: Extract 17 structural features
    Phase 1.5: Pattern detection shortcut (high confidence → return)
    Phase 1.7: Predictor shortcut (confidence > threshold → return)
    Phase 2: Tool matching deep analysis
    Phase 3: Final classification

    Returns:
        dict with 'level' (0-3), 'confidence' ('high'/'medium'/'low'),
        'evidence' (supporting data), 'phase' (where classification happened).
    """
    if predictor is None:
        predictor = _get_predictor()

    evidence = {}

    # --- Phase 1: Structural analysis ---
    features = extract_query_features(query, catalog)
    evidence['features'] = features

    # --- Phase 1.5: Pattern shortcut ---
    patterns = detect_query_patterns(query, catalog)
    evidence['patterns'] = patterns

    if patterns['shortcut_available'] and patterns['shortcut_confidence'] in ('high', 'medium'):
        return {
            'level': patterns['shortcut_level'],
            'confidence': patterns['shortcut_confidence'],
            'evidence': evidence,
            'phase': 'pattern_shortcut',
        }

    # --- Phase 1.7: Predictor shortcut ---
    prediction = predictor.predict(features)
    evidence['prediction'] = prediction

    if prediction['confidence'] == 'high':
        # Cross-validate with pattern if available
        if patterns['shortcut_available']:
            if patterns['shortcut_level'] == prediction['level']:
                return {
                    'level': prediction['level'],
                    'confidence': 'high',
                    'evidence': evidence,
                    'phase': 'predictor_confirmed',
                }
            # Disagreement: trust pattern for SC(0), predictor for higher
            if patterns['shortcut_level'] == 0:
                return {
                    'level': 0,
                    'confidence': 'medium',
                    'evidence': evidence,
                    'phase': 'pattern_override',
                }
        return {
            'level': prediction['level'],
            'confidence': prediction['confidence'],
            'evidence': evidence,
            'phase': 'predictor_shortcut',
        }

    # --- Phase 2: Tool matching deep analysis ---
    tool_analysis = _analyze_tool_matching(query, catalog, features)
    evidence['tool_analysis'] = tool_analysis

    # --- Phase 3: Final classification ---
    return _classify(features, patterns, prediction, tool_analysis, evidence)


def _analyze_tool_matching(
    query: str,
    catalog: ToolCatalog,
    features: Dict,
) -> Dict:
    """Deep analysis of how tools match the query.

    Examines:
    - Uniqueness of tool assignments per sub-task
    - Composability of matched tools
    - Constraint satisfaction complexity
    """
    subtasks = _split_subtasks(query)
    all_keywords = _extract_keywords(query)
    candidates = catalog.find_tools(all_keywords)

    # Per-subtask analysis
    unique_assignments = 0
    ambiguous_assignments = 0
    composable_pairs = 0
    total_pairs = 0

    subtask_details = []
    for st in subtasks:
        kw = _extract_keywords(st)
        matched = catalog.find_tools(kw)
        if len(matched) == 1:
            unique_assignments += 1
        elif len(matched) > 1:
            ambiguous_assignments += 1
        subtask_details.append({
            'text': st[:50],
            'num_tools': len(matched),
            'tools': [t.name for t in matched[:5]],
        })

    # Check composability between consecutive subtask tools
    for i in range(len(subtask_details) - 1):
        t1_tools = subtask_details[i]['tools']
        t2_tools = subtask_details[i + 1]['tools']
        for t1 in t1_tools:
            for t2 in t2_tools:
                total_pairs += 1
                if catalog.can_compose(t1, t2):
                    composable_pairs += 1

    composability_ratio = composable_pairs / max(total_pairs, 1)

    return {
        'num_subtasks': len(subtasks),
        'unique_assignments': unique_assignments,
        'ambiguous_assignments': ambiguous_assignments,
        'composability_ratio': composability_ratio,
        'subtask_details': subtask_details,
        'total_candidates': len(candidates),
    }


def _classify(
    features: Dict,
    patterns: Dict,
    prediction: Dict,
    tool_analysis: Dict,
    evidence: Dict,
) -> Dict:
    """Final SC classification from all collected evidence.

    Decision logic (analogous to _classify in selector-complexity):
    - All subtasks uniquely assigned → SC(0)
    - Clear decomposition, good composability → SC(1)
    - Ambiguous assignments, constraints → SC(2)
    - High overlap, cross-dependencies → SC(3)
    """
    num_subtasks = tool_analysis['num_subtasks']
    unique = tool_analysis['unique_assignments']
    ambiguous = tool_analysis['ambiguous_assignments']
    composability = tool_analysis['composability_ratio']
    total_candidates = tool_analysis['total_candidates']

    # SC(0): Everything maps uniquely
    if num_subtasks <= 1 and total_candidates <= 2 and ambiguous == 0:
        return {
            'level': 0,
            'confidence': 'high',
            'evidence': evidence,
            'phase': 'final_classification',
        }

    # SC(0): Multiple subtasks but all uniquely assigned and composable
    if ambiguous == 0 and unique == num_subtasks and num_subtasks <= 2:
        return {
            'level': 0,
            'confidence': 'medium',
            'evidence': evidence,
            'phase': 'final_classification',
        }

    # SC(1): Clear decomposition with composable tools
    if (unique >= num_subtasks * 0.5
            and composability > 0.3
            and features.get('multi_tool_tasks', 0) == 0):
        confidence = 'high' if composability > 0.6 else 'medium'
        return {
            'level': 1,
            'confidence': confidence,
            'evidence': evidence,
            'phase': 'final_classification',
        }

    # SC(1): Pipeline pattern even with some ambiguity
    if (patterns['shortcut_available']
            and patterns['shortcut_level'] == 1
            and ambiguous <= num_subtasks * 0.5):
        return {
            'level': 1,
            'confidence': 'medium',
            'evidence': evidence,
            'phase': 'final_classification',
        }

    # SC(3): High tool overlap + many dependencies + many subtasks
    if (features.get('tool_overlap', 0) > 0.5
            and features.get('multi_tool_tasks', 0) > 2
            and features.get('max_dependency_depth', 0) > 3):
        return {
            'level': 3,
            'confidence': 'medium',
            'evidence': evidence,
            'phase': 'final_classification',
        }

    # SC(3): Predictor says 3 and topology agrees
    if (prediction['level'] == 3
            and patterns.get('details', {}).get('topology', {})
            and patterns['details'].get('topology', {}).get('level', 0) >= 2):
        return {
            'level': 3,
            'confidence': 'medium',
            'evidence': evidence,
            'phase': 'final_classification',
        }

    # SC(2): Ambiguous with constraints
    if (ambiguous > 0
            and features.get('num_constraints', 0) >= 2):
        return {
            'level': 2,
            'confidence': 'medium',
            'evidence': evidence,
            'phase': 'final_classification',
        }

    # SC(2): Multiple subtasks with tool overlap
    if (num_subtasks >= 3
            and features.get('tool_overlap', 0) > 0.2):
        return {
            'level': 2,
            'confidence': 'low',
            'evidence': evidence,
            'phase': 'final_classification',
        }

    # Fallback to predictor
    return {
        'level': prediction['level'],
        'confidence': 'low',
        'evidence': evidence,
        'phase': 'predictor_fallback',
    }
