"""SC-Router: AI routing based on Selector Complexity theory.

Classifies queries by the difficulty of the routing decision itself,
using the mathematical framework of Selector Complexity (IPS proof complexity).

Quick start:
    >>> from sc_router import ToolCatalog, Tool, route
    >>> catalog = ToolCatalog()
    >>> catalog.register(Tool(
    ...     name="weather",
    ...     description="Get weather forecast",
    ...     input_types={"location"},
    ...     output_types={"weather_data"},
    ...     capability_tags={"weather", "forecast", "temperature"},
    ... ))
    >>> result = route("What's the weather in Madrid?", catalog)
    >>> result.sc_level
    0
    >>> result.strategy
    'direct'
"""

__version__ = "0.1.0"

from .catalog import Tool, ToolCatalog
from .router import RoutingResult, ToolAssignment, route
from .classifier import classify_query
from .features import extract_query_features
from .predictor import SCRouterPredictor
from .decomposer import decompose, DecompositionResult, SubTask
from .patterns import detect_query_patterns
from .adapter import ToolAdapter
from .cost import CostTracker, RoutingRecord

__all__ = [
    'Tool',
    'ToolCatalog',
    'RoutingResult',
    'ToolAssignment',
    'route',
    'classify_query',
    'extract_query_features',
    'SCRouterPredictor',
    'decompose',
    'DecompositionResult',
    'SubTask',
    'detect_query_patterns',
    'ToolAdapter',
    'CostTracker',
    'RoutingRecord',
]
