"""Shared fixtures for SC-Router tests."""

import pytest
from sc_router import Tool, ToolCatalog


@pytest.fixture
def sample_catalog():
    """A catalog with 10 tools covering various capabilities."""
    catalog = ToolCatalog()

    catalog.register(Tool(
        name="weather",
        description="Get weather forecast for a location",
        input_types={"location"},
        output_types={"weather_data"},
        capability_tags={"weather", "forecast", "temperature", "climate"},
    ))
    catalog.register(Tool(
        name="calculator",
        description="Perform arithmetic calculations",
        input_types={"expression"},
        output_types={"number"},
        capability_tags={"math", "calculate", "arithmetic", "number"},
    ))
    catalog.register(Tool(
        name="search",
        description="Search the web for information",
        input_types={"query"},
        output_types={"search_results"},
        capability_tags={"search", "web", "find", "lookup", "information"},
    ))
    catalog.register(Tool(
        name="summarizer",
        description="Summarize text content",
        input_types={"text", "search_results"},
        output_types={"summary"},
        capability_tags={"summarize", "summary", "condense", "text"},
    ))
    catalog.register(Tool(
        name="translator",
        description="Translate text between languages",
        input_types={"text", "summary"},
        output_types={"translated_text"},
        capability_tags={"translate", "language", "translation"},
    ))
    catalog.register(Tool(
        name="flight_search",
        description="Search for flights between cities",
        input_types={"origin", "destination", "date"},
        output_types={"flight_list"},
        capability_tags={"flight", "flights", "travel", "book", "airline"},
    ))
    catalog.register(Tool(
        name="hotel_search",
        description="Search for hotels in a city",
        input_types={"city", "date"},
        output_types={"hotel_list"},
        capability_tags={"hotel", "hotels", "accommodation", "travel", "book"},
    ))
    catalog.register(Tool(
        name="restaurant_search",
        description="Find restaurants in a location",
        input_types={"location"},
        output_types={"restaurant_list"},
        capability_tags={"restaurant", "restaurants", "food", "dining"},
    ))
    catalog.register(Tool(
        name="budget_optimizer",
        description="Optimize spending within a budget constraint",
        input_types={"flight_list", "hotel_list", "restaurant_list", "number"},
        output_types={"optimized_plan"},
        capability_tags={"budget", "optimize", "cost", "plan", "spending"},
    ))
    catalog.register(Tool(
        name="sentiment_analyzer",
        description="Analyze sentiment of text and social media data",
        input_types={"text", "search_results"},
        output_types={"sentiment_data"},
        capability_tags={"sentiment", "analyze", "social", "opinion", "trend"},
    ))

    return catalog
