"""Basic SC-Router usage example."""

from sc_router import ToolCatalog, Tool, route, classify_query

# --- Build a catalog ---
catalog = ToolCatalog()

catalog.register(Tool(
    name="weather",
    description="Get weather forecast for a location",
    input_types={"location"},
    output_types={"weather_data"},
    capability_tags={"weather", "forecast", "temperature"},
))
catalog.register(Tool(
    name="calculator",
    description="Perform arithmetic calculations",
    input_types={"expression"},
    output_types={"number"},
    capability_tags={"math", "calculate", "arithmetic"},
))
catalog.register(Tool(
    name="search",
    description="Search the web for information",
    input_types={"query"},
    output_types={"search_results"},
    capability_tags={"search", "web", "find", "information"},
))
catalog.register(Tool(
    name="summarizer",
    description="Summarize text content",
    input_types={"text", "search_results"},
    output_types={"summary"},
    capability_tags={"summarize", "summary", "condense"},
))

# --- Test queries at each SC level ---
queries = [
    ("SC(0) - Direct", "What's the weather in Madrid?"),
    ("SC(0) - Direct", "Calculate 15 * 37"),
    ("SC(1) - Pipeline", "Search the web for AI news, then summarize the results"),
    ("SC(2) - Complex", "Search for information about climate change, summarize it, "
                        "and calculate the economic impact within a budget of $1M"),
]

print("=" * 70)
print("SC-Router: Basic Routing Example")
print("=" * 70)

for label, query in queries:
    result = route(query, catalog)
    print(f"\n[{label}] {query}")
    print(f"  SC Level:  {result.sc_level}")
    print(f"  Strategy:  {result.strategy}")
    print(f"  Tools:     {[a.tool for a in result.tool_assignments]}")
    print(f"  Phase:     {result.classification.get('phase', 'N/A')}")
