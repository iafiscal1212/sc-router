# SC-Router

AI routing based on Selector Complexity theory.

**"What is the minimum cost of choosing the right strategy?"**

SC-Router classifies queries by the difficulty of the routing decision itself — not just the query content. It determines whether a query needs direct dispatch, pipeline decomposition, combinatorial search, or full agent delegation.

**v0.3.0** adds distributed execution: register remote agents, execute across microservices, health checks with circuit breaker, and an optional HTTP gateway — all without touching the core classifier (<0.5ms p99).

Part of [**kore-stack**](https://github.com/iafiscal1212/kore-stack) — the complete cognitive middleware for LLMs. `pip install kore-stack` for the full stack, or install individually:

## Install

```bash
pip install sc-router              # core router (zero dependencies)
pip install sc-router[gateway]     # + HTTP gateway (starlette, uvicorn, httpx, pyyaml)
pip install kore-bridge[sc]        # integrated with kore-bridge
pip install kore-stack             # full stack: mind + bridge + SC routing
```

## Quick Start

### Local routing (as before)

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

### Distributed routing (new in v0.3.0)

```python
import asyncio
from sc_router import RemoteAgent, AgentRegistry, AgentStatus, route
from sc_router.executor import execute
from sc_router.catalog import Tool

# 1. Register remote agents
registry = AgentRegistry()
registry.register(RemoteAgent(
    id="search-agent",
    url="http://search-service:8081",
    tool=Tool(
        name="search",
        description="Search the web",
        input_types={"query"},
        output_types={"search_results"},
        capability_tags={"search", "web", "find"},
    ),
    status=AgentStatus.HEALTHY,
))

# 2. Classify (still <50ms, zero overhead)
result = route("Search for Python tutorials", registry.catalog)

# 3. Execute against remote agents
exec_result = asyncio.run(execute(result, registry))
print(exec_result.outputs)
```

### YAML config + Gateway

```yaml
# config.yaml
agents:
  - id: search-agent
    url: http://search:8081
    tool:
      name: search
      description: "Search the web"
      capability_tags: [search, web]
      input_types: [query]
      output_types: [search_results]

  - id: weather-agent
    url: http://weather:8082
    tool:
      name: weather
      description: "Get weather forecast"
      capability_tags: [weather, forecast]
      input_types: [location]
      output_types: [weather_data]

health:
  failure_threshold: 3
  recovery_timeout_s: 30
```

```bash
# Start the gateway
python -m uvicorn sc_router.gateway:create_app --factory --host 0.0.0.0 --port 8080
```

```bash
# Classify + execute
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Madrid?"}'

# Classify only (no remote calls)
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Madrid?", "execute": false}'

# Health check
curl http://localhost:8080/health

# List agents
curl http://localhost:8080/agents
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

The classification runs in **<0.5ms p99** and adds zero overhead to any routing pipeline.

## Architecture (v0.3.0)

```
                          ┌─────────────────┐
       POST /route ──────►│    Gateway       │
                          │  (Starlette)     │
                          └────────┬─────────┘
                                   │
                     ┌─────────────▼──────────────┐
                     │     SC Classification       │
                     │   17 features, <0.5ms p99   │
                     └─────────────┬──────────────┘
                                   │
                     ┌─────────────▼──────────────┐
                     │    AgentRegistry            │
                     │  RemoteAgent + ToolCatalog   │
                     │  Health checks + Circuit     │
                     │  breaker                     │
                     └─────────────┬──────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                   │
         ┌──────▼──────┐  ┌───────▼──────┐  ┌────────▼──────┐
         │ Agent A      │  │ Agent B      │  │ Agent C       │
         │ (search)     │  │ (weather)    │  │ (summarizer)  │
         └─────────────┘  └──────────────┘  └───────────────┘
```

**Core** (`pip install sc-router`): zero dependencies, classification + local routing.

**Gateway** (`pip install sc-router[gateway]`): Starlette HTTP gateway, distributed execution, health checks, YAML config.

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

## Performance

Benchmarked on 10-tool catalog, 100 iterations per query (v0.3.0):

| Query | avg | p50 | p95 | p99 |
|---|---|---|---|---|
| SC(0) direct | 0.08ms | 0.07ms | 0.13ms | 0.18ms |
| SC(1) pipeline | 0.27ms | 0.26ms | 0.41ms | 0.44ms |
| SC(2) constrained | 0.47ms | 0.46ms | 0.56ms | 0.59ms |
| SC(3) entangled | 0.22ms | 0.23ms | 0.32ms | 0.44ms |

Distributed layer adds **zero overhead** to classification. Scales to 50+ tools under 50ms.

## Part of kore-stack

| Package | What it does |
|---------|-------------|
| [kore-mind](https://github.com/iafiscal1212/kore-mind) | Memory, identity, traces, cache storage |
| [kore-bridge](https://github.com/iafiscal1212/kore-bridge) | LLM integration, cache logic, rate limiting, A/B testing, SC routing |
| **sc-router** (this) | Query routing by Selector Complexity theory |
| [**kore-stack**](https://github.com/iafiscal1212/kore-stack) | All of the above, one install: `pip install kore-stack` |

## License

Copyright (c) 2024-2026 Carmen Esteban. All rights reserved. No part of this software may be copied, modified, distributed or used without express written permission.
