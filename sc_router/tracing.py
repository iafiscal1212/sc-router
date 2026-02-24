"""Observability and tracing for distributed routing.

RoutingTrace captures: query, sc_level, classification_ms, execution_ms, steps.
TracingHook provides pluggable listeners (for OTEL, kore-mind, or custom).
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TraceStep:
    """A single step in a routing execution trace."""
    agent_id: str
    tool_name: str
    subtask: str
    status: str = 'pending'  # pending, running, success, error, skipped
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None

    def start(self) -> None:
        self.status = 'running'
        self.started_at = time.monotonic()

    def finish(self, result: Any = None, error: Optional[str] = None) -> None:
        self.finished_at = time.monotonic()
        self.duration_ms = (self.finished_at - self.started_at) * 1000
        if error:
            self.status = 'error'
            self.error = error
        else:
            self.status = 'success'
            self.result = result

    def skip(self, reason: str = 'agent unhealthy') -> None:
        self.status = 'skipped'
        self.error = reason


@dataclass
class RoutingTrace:
    """Complete trace of a routing decision + execution.

    Captures timing, SC classification, and per-step details.
    """
    query: str
    sc_level: int = -1
    strategy: str = ''
    confidence: str = ''
    classification_ms: float = 0.0
    execution_ms: float = 0.0
    total_ms: float = 0.0
    steps: List[TraceStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    _start_time: float = field(default=0.0, repr=False)

    def start_classification(self) -> None:
        self._start_time = time.monotonic()

    def finish_classification(self, sc_level: int, strategy: str,
                              confidence: str = '') -> None:
        self.classification_ms = (time.monotonic() - self._start_time) * 1000
        self.sc_level = sc_level
        self.strategy = strategy
        self.confidence = confidence

    def start_execution(self) -> None:
        self._start_time = time.monotonic()

    def finish_execution(self) -> None:
        self.execution_ms = (time.monotonic() - self._start_time) * 1000
        self.total_ms = self.classification_ms + self.execution_ms

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    @property
    def success(self) -> bool:
        """True if all non-skipped steps succeeded."""
        active = [s for s in self.steps if s.status != 'skipped']
        return bool(active) and all(s.status == 'success' for s in active)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses / logging."""
        return {
            'query': self.query,
            'sc_level': self.sc_level,
            'strategy': self.strategy,
            'confidence': self.confidence,
            'classification_ms': round(self.classification_ms, 2),
            'execution_ms': round(self.execution_ms, 2),
            'total_ms': round(self.total_ms, 2),
            'success': self.success,
            'steps': [
                {
                    'agent_id': s.agent_id,
                    'tool_name': s.tool_name,
                    'subtask': s.subtask,
                    'status': s.status,
                    'duration_ms': round(s.duration_ms, 2),
                    'error': s.error,
                }
                for s in self.steps
            ],
            'metadata': self.metadata,
            'timestamp': self.timestamp,
        }

    def to_kore_mind_trace(self) -> Dict[str, Any]:
        """Convert to kore-mind compatible trace format."""
        return {
            'type': 'sc_routing',
            'query': self.query,
            'sc_level': self.sc_level,
            'strategy': self.strategy,
            'timing': {
                'classification_ms': round(self.classification_ms, 2),
                'execution_ms': round(self.execution_ms, 2),
                'total_ms': round(self.total_ms, 2),
            },
            'steps': [
                {
                    'agent': s.agent_id,
                    'tool': s.tool_name,
                    'task': s.subtask,
                    'status': s.status,
                    'latency_ms': round(s.duration_ms, 2),
                }
                for s in self.steps
            ],
            'success': self.success,
            'timestamp': self.timestamp,
        }


# Type alias for listener callbacks
TraceListener = Callable[[RoutingTrace], None]


class TracingHook:
    """Pluggable tracing hook with listener support.

    Register listeners that get called when a trace completes.
    Compatible with OTEL exporters, kore-mind, or custom loggers.

    Usage:
        hook = TracingHook()
        hook.add_listener(lambda trace: print(trace.to_dict()))
        # ... later, after routing + execution:
        hook.record(trace)
    """

    def __init__(self) -> None:
        self._listeners: List[TraceListener] = []
        self._traces: List[RoutingTrace] = []
        self._max_traces: int = 1000

    def add_listener(self, listener: TraceListener) -> None:
        """Register a trace listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener: TraceListener) -> None:
        """Remove a trace listener."""
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def record(self, trace: RoutingTrace) -> None:
        """Record a completed trace and notify all listeners."""
        self._traces.append(trace)
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces:]
        for listener in self._listeners:
            listener(trace)

    @property
    def traces(self) -> List[RoutingTrace]:
        """All recorded traces."""
        return list(self._traces)

    @property
    def count(self) -> int:
        return len(self._traces)

    def clear(self) -> None:
        """Clear all recorded traces."""
        self._traces.clear()
