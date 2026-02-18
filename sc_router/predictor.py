"""Threshold-based decision tree for SC level prediction.

Analogous to SCPredictor in selector-complexity/predictor.py.
No ML dependencies — uses threshold splits on the 17 features.
Training data: ~40 labeled queries with known SC levels.
"""

from typing import Dict, List, Optional, Tuple

from .features import FEATURE_NAMES


class SCRouterPredictor:
    """Predict SC level (0-3) from query features using threshold splits.

    Same algorithm as SCPredictor in selector-complexity:
    - For each feature, find the best threshold that separates SC levels.
    - At prediction time, each split votes (weighted by accuracy).
    - Fallback: nearest-mean classification.
    """

    def __init__(self):
        self._splits: List[Tuple[str, float, int, int, float]] = []
        # (feature_name, threshold, level_below, level_above, accuracy)
        self._level_means: Dict[int, Dict[str, float]] = {}
        self._fitted = False

    def fit(self, training_data: List[Tuple[Dict[str, float], int]]) -> 'SCRouterPredictor':
        """Fit from labeled (features_dict, sc_level) pairs.

        For each feature, tries every midpoint threshold between sorted unique
        values and picks the split with highest classification accuracy.
        Keeps splits with accuracy > 0.6.
        """
        if not training_data:
            return self

        # Group by level
        by_level: Dict[int, List[Dict[str, float]]] = {}
        for features, level in training_data:
            by_level.setdefault(level, []).append(features)

        # Compute per-level feature means (for fallback)
        self._level_means = {}
        for level, samples in by_level.items():
            means = {}
            for fname in FEATURE_NAMES:
                vals = [s.get(fname, 0.0) for s in samples]
                means[fname] = sum(vals) / max(len(vals), 1)
            self._level_means[level] = means

        # Find best threshold splits
        self._splits = []
        all_levels = sorted(by_level.keys())

        for fname in FEATURE_NAMES:
            # Collect (value, level) pairs
            pairs = []
            for features, level in training_data:
                pairs.append((features.get(fname, 0.0), level))
            pairs.sort(key=lambda x: x[0])

            # Try midpoint thresholds
            values = sorted(set(v for v, _ in pairs))
            if len(values) < 2:
                continue

            best_acc = 0.0
            best_split = None

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2.0

                # For each pair of levels, try this threshold
                for l_below in all_levels:
                    for l_above in all_levels:
                        if l_below == l_above:
                            continue
                        correct = 0
                        total = 0
                        for val, level in pairs:
                            if level == l_below and val <= threshold:
                                correct += 1
                                total += 1
                            elif level == l_above and val > threshold:
                                correct += 1
                                total += 1
                            elif level in (l_below, l_above):
                                total += 1
                        if total > 0:
                            acc = correct / total
                            if acc > best_acc:
                                best_acc = acc
                                best_split = (fname, threshold, l_below, l_above, acc)

            if best_split and best_acc > 0.6:
                self._splits.append(best_split)

        # Sort by accuracy descending
        self._splits.sort(key=lambda x: -x[4])
        self._fitted = True
        return self

    def predict(self, features: Dict[str, float]) -> Dict:
        """Predict SC level from features.

        Returns dict with 'level', 'confidence', 'method'.
        """
        if not self._fitted:
            return {'level': 2, 'confidence': 'low', 'method': 'unfitted'}

        # Weighted voting from splits
        votes: Dict[int, float] = {}
        for fname, threshold, l_below, l_above, acc in self._splits:
            val = features.get(fname, 0.0)
            predicted = l_below if val <= threshold else l_above
            votes[predicted] = votes.get(predicted, 0.0) + acc

        if votes:
            total_weight = sum(votes.values())
            winner = max(votes, key=votes.get)
            winner_weight = votes[winner]
            ratio = winner_weight / total_weight if total_weight > 0 else 0

            if ratio > 0.7:
                confidence = 'high'
            elif ratio > 0.5:
                confidence = 'medium'
            else:
                confidence = 'low'

            return {'level': winner, 'confidence': confidence, 'method': 'threshold_voting'}

        # Fallback: nearest-mean
        return self._nearest_mean(features)

    def _nearest_mean(self, features: Dict[str, float]) -> Dict:
        """Classify by L2 distance to level means."""
        best_level = 0
        best_dist = float('inf')

        for level, means in self._level_means.items():
            dist = 0.0
            for fname in FEATURE_NAMES:
                diff = features.get(fname, 0.0) - means.get(fname, 0.0)
                dist += diff * diff
            if dist < best_dist:
                best_dist = dist
                best_level = level

        return {'level': best_level, 'confidence': 'low', 'method': 'nearest_mean'}

    def fit_from_examples(self) -> 'SCRouterPredictor':
        """Auto-fit from built-in training examples.

        ~40 queries covering SC(0) through SC(3), with pre-computed feature
        signatures. Analogous to fit_from_landscape() in selector-complexity.
        """
        training_data = _get_training_data()
        return self.fit(training_data)


def _get_training_data() -> List[Tuple[Dict[str, float], int]]:
    """Built-in training dataset of ~40 labeled query feature signatures."""
    data = []

    # --- SC(0): Single tool, obvious dispatch ---
    # Pattern: low num_candidates, 1 subtask, low constraints
    for i in range(10):
        nc = 0.0 + (i % 3) * 0.5  # 0-1 constraints
        nt = 1.0 + (i % 2)        # 1-2 candidate tools
        data.append(({
            'num_constraints': nc,
            'num_candidate_tools': nt,
            'max_dependency_depth': 1.0,
            'avg_dependency_depth': 1.0,
            'tool_coverage': 0.05 + i * 0.01,
            'tool_overlap': 0.0,
            'query_expansion': 1.0,
            'constraint_tool_ratio': nc / max(nt, 1),
            'single_tool_tasks': 1.0,
            'two_tool_tasks': 0.0,
            'multi_tool_tasks': 0.0,
            'avg_task_breadth': nt,
            'max_task_breadth': nt,
            'tool_clusters': 1.0,
            'max_cluster_size': nt,
            'spectral_gap': 0.0,
            'avg_tool_specificity': 0.8 + i * 0.01,
        }, 0))

    # --- SC(1): Decomposable pipeline ---
    # Pattern: 2-3 subtasks, sequential composability, moderate candidates
    for i in range(10):
        nsub = 2.0 + (i % 2)
        nt = 2.0 + i * 0.3
        data.append(({
            'num_constraints': 1.0 + (i % 3),
            'num_candidate_tools': nt,
            'max_dependency_depth': nsub,
            'avg_dependency_depth': nsub * 0.7,
            'tool_coverage': 0.1 + i * 0.02,
            'tool_overlap': 0.1 + i * 0.03,
            'query_expansion': nsub,
            'constraint_tool_ratio': (1.0 + i % 3) / max(nt, 1),
            'single_tool_tasks': nsub * 0.6,
            'two_tool_tasks': nsub * 0.3,
            'multi_tool_tasks': 0.0,
            'avg_task_breadth': 1.3 + i * 0.05,
            'max_task_breadth': 2.0,
            'tool_clusters': 1.0 + (i % 2),
            'max_cluster_size': nt * 0.7,
            'spectral_gap': 0.3 + i * 0.05,
            'avg_tool_specificity': 0.6 + i * 0.02,
        }, 1))

    # --- SC(2): Ambiguous/complex, search needed ---
    # Pattern: multiple subtasks, constraints, overlapping tools
    for i in range(10):
        nsub = 3.0 + i * 0.5
        nt = 4.0 + i * 0.5
        nc = 2.0 + i * 0.5
        data.append(({
            'num_constraints': nc,
            'num_candidate_tools': nt,
            'max_dependency_depth': 2.0 + i * 0.3,
            'avg_dependency_depth': 1.5 + i * 0.2,
            'tool_coverage': 0.2 + i * 0.03,
            'tool_overlap': 0.3 + i * 0.04,
            'query_expansion': nsub * 0.8,
            'constraint_tool_ratio': nc / max(nt, 1),
            'single_tool_tasks': 1.0,
            'two_tool_tasks': nsub * 0.4,
            'multi_tool_tasks': nsub * 0.3,
            'avg_task_breadth': 2.5 + i * 0.2,
            'max_task_breadth': 4.0 + i * 0.3,
            'tool_clusters': 2.0 + (i % 3),
            'max_cluster_size': nt * 0.5,
            'spectral_gap': 0.5 + i * 0.04,
            'avg_tool_specificity': 0.4 + i * 0.03,
        }, 2))

    # --- SC(3): Globally entangled, full agent ---
    # Pattern: many subtasks, high overlap, deep dependencies
    for i in range(10):
        nsub = 4.0 + i
        nt = 6.0 + i
        nc = 3.0 + i
        data.append(({
            'num_constraints': nc,
            'num_candidate_tools': nt,
            'max_dependency_depth': 3.0 + i * 0.5,
            'avg_dependency_depth': 2.5 + i * 0.4,
            'tool_coverage': 0.4 + i * 0.04,
            'tool_overlap': 0.5 + i * 0.04,
            'query_expansion': nsub,
            'constraint_tool_ratio': nc / max(nt, 1),
            'single_tool_tasks': 0.0,
            'two_tool_tasks': 1.0,
            'multi_tool_tasks': nsub * 0.6,
            'avg_task_breadth': 3.5 + i * 0.3,
            'max_task_breadth': 6.0 + i * 0.5,
            'tool_clusters': 1.0,
            'max_cluster_size': nt,
            'spectral_gap': 0.8 + i * 0.02,
            'avg_tool_specificity': 0.2 + i * 0.02,
        }, 3))

    return data
