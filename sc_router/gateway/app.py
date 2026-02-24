"""Starlette ASGI app factory for the sc-router gateway."""

from typing import Any, Dict, Optional

try:
    from starlette.applications import Starlette
    from starlette.routing import Route
except ImportError:
    raise ImportError(
        "starlette is required for the gateway. "
        "Install with: pip install sc-router[gateway]"
    )

from ..agent import AgentRegistry
from ..config import build_registry, load_config
from ..health import HealthChecker
from ..tracing import TracingHook
from .handlers import make_route_handler, make_health_handler, make_agents_handler


def create_app(
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    registry: Optional[AgentRegistry] = None,
) -> Starlette:
    """Create a Starlette ASGI app for the sc-router gateway.

    Provide one of:
      - config_path: Path to a YAML config file
      - config: A parsed config dict
      - registry: A pre-built AgentRegistry

    Returns:
        A Starlette app with POST /route, GET /health, GET /agents.
    """
    if registry is None:
        if config is None:
            if config_path is None:
                raise ValueError(
                    "Provide config_path, config dict, or registry")
            config = load_config(config_path)
        registry = build_registry(config)

    # Health config
    health_cfg = {}
    if config:
        health_cfg = config.get('health', {})

    health_checker = HealthChecker(
        registry,
        failure_threshold=health_cfg.get('failure_threshold', 3),
        recovery_timeout=health_cfg.get('recovery_timeout_s', 30),
        check_timeout=health_cfg.get('check_timeout_s', 5),
    )
    tracing_hook = TracingHook()

    routes = [
        Route('/route', make_route_handler(registry, health_checker, tracing_hook),
              methods=['POST']),
        Route('/health', make_health_handler(registry, health_checker),
              methods=['GET']),
        Route('/agents', make_agents_handler(registry),
              methods=['GET']),
    ]

    app = Starlette(routes=routes)

    # Store references on app state for testing / access
    app.state.registry = registry
    app.state.health_checker = health_checker
    app.state.tracing_hook = tracing_hook

    return app
