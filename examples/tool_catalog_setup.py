"""Example: Building a tool catalog from various sources."""

from sc_router import ToolCatalog, Tool, ToolAdapter, route

# --- Method 1: Direct registration ---
catalog = ToolCatalog()
catalog.register(Tool(
    name="weather",
    description="Get weather forecast",
    input_types={"location"},
    output_types={"weather_data"},
    capability_tags={"weather", "forecast", "temperature"},
))

# --- Method 2: From a Python function ---
def search_web(query: str) -> dict:
    """Search the web for information and return results."""
    pass

catalog.register(ToolAdapter.from_function(search_web))

# --- Method 3: From a description ---
catalog.register(ToolAdapter.from_description(
    "summarizer",
    "Summarize long text content into concise bullet points",
))

# --- Method 4: From OpenAI function definitions ---
openai_functions = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                },
                "required": ["ticker"],
            },
        },
    },
]

from sc_router.integrations.openai import from_openai_functions
for tool in from_openai_functions(openai_functions):
    catalog.register(tool)

# --- Show catalog ---
print(f"Catalog size: {catalog.size} tools")
for tool in catalog.tools:
    print(f"  - {tool.name}: {tool.description}")
    print(f"    Tags: {tool.capability_tags}")

# --- Route ---
print("\n--- Routing ---")
result = route("What's the weather in London?", catalog)
print(f"Query: 'What's the weather in London?'")
print(f"  SC Level: {result.sc_level}")
print(f"  Tool: {result.tool_assignments[0].tool if result.tool_assignments else 'none'}")
