# SC-Router

AI routing based on Selector Complexity theory.

**"What is the minimum cost of choosing the right strategy?"**

SC-Router classifies queries by the difficulty of the routing decision itself — not just the query content. Based on the mathematical framework of [Selector Complexity](https://pypi.org/project/selector-complexity/) (IPS proof complexity), it determines whether a query needs direct dispatch, pipeline decomposition, combinatorial search, or full agent delegation.

## Install

```bash
pip install sc-router
```

## Quick Start

```python
from sc_router import ToolCatalog, Tool, route

catalog = ToolCatalog()
catalog.register(Tool(
    name="weather",
    description="Get weather forecast",
    input_types={"location"},
    output_types={"weather_data"},
    capability_tags={"weather", "forecast", "temperature"}
))
catalog.register(Tool(
    name="calculator",
    description="Perform arithmetic calculations",
    input_types={"expression"},
    output_types={"number"},
    capability_tags={"math", "calculate", "arithmetic"}
))

result = route("What's the weather in Madrid?", catalog)
print(result.sc_level)           # 0
print(result.strategy)           # 'direct'
print(result.tool_assignments)   # [ToolAssignment(tool='weather', ...)]
```

## SC Levels

| SC | Query Type | Routing Action | Example |
|---|---|---|---|
| **SC(0)** | 1 tool, obvious | Direct dispatch | "What's the weather in Madrid?" |
| **SC(1)** | Decomposable | Pipeline/parallel | "Search flights to Paris, book the cheapest" |
| **SC(2)** | Ambiguous/complex | Search combinations | "Plan trip: flights+hotel+restaurants, budget $2000" |
| **SC(3)** | Globally entangled | Full agent | "Analyze market trends, cross with social sentiment, build predictive model" |

## How It Works

SC-Router extracts 17 structural features from each query (analogous to the 17 features in IPS proof complexity), then classifies the routing difficulty using a threshold-based decision tree — no ML required.

The classification runs in <50ms and adds minimal overhead to any routing pipeline.

## License

MIT
