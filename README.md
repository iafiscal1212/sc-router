# SC-Router

AI routing based on Selector Complexity theory.

**"What is the minimum cost of choosing the right strategy?"**

SC-Router classifies queries by the difficulty of the routing decision itself — not just the query content. It determines whether a query needs direct dispatch, pipeline decomposition, combinatorial search, or full agent delegation.

Part of [**kore-stack**](https://github.com/iafiscal1212/kore-stack) — the complete cognitive middleware for LLMs. `pip install kore-stack` for the full stack, or install individually:

## Install

```bash
pip install sc-router          # just the router
pip install kore-bridge[sc]    # integrated with kore-bridge
pip install kore-stack         # full stack: mind + bridge + SC routing
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

SC-Router extracts 17 structural features from each query, then classifies the routing difficulty using a threshold-based decision tree — no ML required.

The classification runs in <50ms and adds minimal overhead to any routing pipeline.

## Integration with kore-bridge

SC-Router plugs directly into kore-bridge as `SCRouterProvider`:

```python
from kore_bridge import SCRouterProvider, Bridge, OllamaProvider
from kore_bridge.providers import OpenAIProvider
from kore_mind import Mind
from sc_router import ToolCatalog, Tool

catalog = ToolCatalog()
catalog.register(Tool(
    name="calculator",
    description="Arithmetic calculations",
    input_types={"expression"},
    output_types={"number"},
    capability_tags={"math", "calculate"},
))

router = SCRouterProvider(
    providers={
        "fast": OllamaProvider(model="llama3.2"),
        "quality": OpenAIProvider(model="gpt-4o"),
    },
    catalog=catalog,
)

bridge = Bridge(mind=Mind("agent.db"), llm=router)
bridge.think("What is 2+2?")          # SC(0) → Ollama
print(router.last_sc_level)           # 0
```

## Part of kore-stack

| Package | What it does |
|---------|-------------|
| [kore-mind](https://github.com/iafiscal1212/kore-mind) | Memory, identity, traces, cache storage |
| [kore-bridge](https://github.com/iafiscal1212/kore-bridge) | LLM integration, cache logic, rate limiting, A/B testing, SC routing |
| **sc-router** (this) | Query routing by Selector Complexity theory |
| [**kore-stack**](https://github.com/iafiscal1212/kore-stack) | All of the above, one install: `pip install kore-stack` |

## License

MIT
