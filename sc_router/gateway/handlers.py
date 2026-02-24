"""HTTP handlers for the sc-router gateway."""

from typing import Callable

try:
    from starlette.requests import Request
    from starlette.responses import JSONResponse
except ImportError:
    raise ImportError(
        "starlette is required for the gateway. "
        "Install with: pip install sc-router[gateway]"
    )

from ..agent import AgentRegistry
from ..executor import execute, ExecutionResult
from ..health import HealthChecker
from ..router import route
from ..tracing import RoutingTrace, TracingHook


def make_route_handler(
    registry: AgentRegistry,
    health_checker: HealthChecker,
    tracing_hook: TracingHook,
) -> Callable:
    """Create the POST /route handler.

    Request body:
        {"query": "...", "execute": true}

    If execute=false, only classification is returned (no remote calls).
    """

    async def handle_route(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {'error': 'Invalid JSON body'}, status_code=400)

        query = body.get('query', '').strip()
        if not query:
            return JSONResponse(
                {'error': 'Missing "query" field'}, status_code=400)

        should_execute = body.get('execute', True)

        # Create trace
        trace = RoutingTrace(query=query)

        # Classify
        trace.start_classification()
        catalog = registry.healthy_catalog()
        if catalog.size == 0:
            catalog = registry.catalog

        routing_result = route(query, catalog)
        trace.finish_classification(
            sc_level=routing_result.sc_level,
            strategy=routing_result.strategy,
            confidence=routing_result.classification.get('confidence', ''),
        )

        response_data = {
            'query': query,
            'sc_level': routing_result.sc_level,
            'strategy': routing_result.strategy,
            'confidence': routing_result.classification.get('confidence', ''),
            'tool_assignments': [
                {
                    'tool': ta.tool,
                    'subtask': ta.subtask,
                    'confidence': ta.confidence,
                }
                for ta in routing_result.tool_assignments
            ],
        }

        if should_execute and routing_result.tool_assignments:
            exec_result = await execute(
                routing_result, registry, trace=trace)
            response_data['execution'] = {
                'success': exec_result.success,
                'outputs': exec_result.outputs,
            }

        response_data['trace'] = trace.to_dict()
        tracing_hook.record(trace)

        return JSONResponse(response_data)

    return handle_route


def make_health_handler(
    registry: AgentRegistry,
    health_checker: HealthChecker,
) -> Callable:
    """Create the GET /health handler."""

    async def handle_health(request: Request) -> JSONResponse:
        agent_status = health_checker.get_status_summary()
        total = registry.size
        healthy = len(registry.healthy_agents())

        return JSONResponse({
            'status': 'healthy' if healthy > 0 or total == 0 else 'degraded',
            'agents_total': total,
            'agents_healthy': healthy,
            'agents': agent_status,
        })

    return handle_health


def make_agents_handler(registry: AgentRegistry) -> Callable:
    """Create the GET /agents handler."""

    async def handle_agents(request: Request) -> JSONResponse:
        return JSONResponse({
            'count': registry.size,
            'agents': registry.to_dict(),
        })

    return handle_agents
