"""Cost tracking and feedback loop.

Tracks actual routing costs, validates SC predictions vs reality,
and feeds back into the predictor for continuous improvement.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .features import FEATURE_NAMES


@dataclass
class RoutingRecord:
    """Record of a single routing decision and its outcome."""
    query: str
    predicted_level: int
    actual_level: Optional[int] = None  # filled after execution
    strategy: str = ''
    tools_used: List[str] = field(default_factory=list)
    latency_ms: float = 0.0  # routing decision latency
    execution_ms: float = 0.0  # total execution time
    success: bool = False
    features: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """Track routing costs and provide feedback for predictor improvement.

    Computes a composite difficulty score and validates predictions.
    """

    def __init__(self, max_history: int = 1000):
        self._history: List[RoutingRecord] = []
        self._max_history = max_history

    def record(self, record: RoutingRecord) -> None:
        """Add a routing record."""
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    @property
    def history(self) -> List[RoutingRecord]:
        return list(self._history)

    def accuracy(self) -> Dict:
        """Compute prediction accuracy from records with known actual levels.

        Returns dict with overall accuracy and per-level breakdown.
        """
        labeled = [r for r in self._history if r.actual_level is not None]
        if not labeled:
            return {'overall': 0.0, 'total': 0, 'per_level': {}}

        correct = sum(1 for r in labeled if r.predicted_level == r.actual_level)
        overall = correct / len(labeled)

        # Per-level
        per_level: Dict[int, Dict] = {}
        for level in range(4):
            level_records = [r for r in labeled if r.actual_level == level]
            if level_records:
                level_correct = sum(1 for r in level_records
                                    if r.predicted_level == r.actual_level)
                per_level[level] = {
                    'accuracy': level_correct / len(level_records),
                    'total': len(level_records),
                    'correct': level_correct,
                }

        return {
            'overall': overall,
            'total': len(labeled),
            'correct': correct,
            'per_level': per_level,
        }

    def difficulty_score(self, record: RoutingRecord) -> float:
        """Compute composite difficulty score (0-100) for a routing record.

        Weighted components:
          - SC level: 40%
          - Tool count: 20%
          - Constraint density: 20%
          - Execution complexity: 20%
        """
        # SC level component (0-40)
        level_scores = {0: 0, 1: 13, 2: 27, 3: 40}
        sc_score = level_scores.get(record.predicted_level, 20)

        # Tool count component (0-20)
        num_tools = len(record.tools_used)
        tool_score = min(num_tools * 4.0, 20.0)

        # Constraint density from features (0-20)
        constraints = record.features.get('num_constraints', 0)
        candidates = record.features.get('num_candidate_tools', 1)
        constraint_density = constraints / max(candidates, 1)
        constraint_score = min(constraint_density * 10.0, 20.0)

        # Execution complexity (0-20)
        overlap = record.features.get('tool_overlap', 0)
        depth = record.features.get('max_dependency_depth', 1)
        exec_score = min((overlap * 10 + depth * 3), 20.0)

        return sc_score + tool_score + constraint_score + exec_score

    def get_feedback_data(self) -> List[Tuple[Dict[str, float], int]]:
        """Extract labeled training data from history for predictor retraining.

        Returns list of (features, actual_level) tuples from records
        where actual_level is known.
        """
        data = []
        for r in self._history:
            if r.actual_level is not None and r.features:
                data.append((r.features, r.actual_level))
        return data

    def summary(self) -> Dict:
        """Summary statistics of routing history."""
        if not self._history:
            return {'total': 0}

        levels = [r.predicted_level for r in self._history]
        latencies = [r.latency_ms for r in self._history if r.latency_ms > 0]
        successes = [r for r in self._history if r.success]

        return {
            'total': len(self._history),
            'success_rate': len(successes) / len(self._history) if self._history else 0,
            'level_distribution': {
                l: levels.count(l) for l in range(4)
            },
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'max_latency_ms': max(latencies) if latencies else 0,
            'prediction_accuracy': self.accuracy(),
        }
