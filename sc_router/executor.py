"""Distributed execution engine for sc-router.

Takes a RoutingResult (already computed in <50ms) and executes it against
remote agents via HTTP.

Strategies:
  - single: 1 HTTP call to a single agent
  - sequential: pipeline — output of A feeds as context to B
  - parallel: fan-out with asyncio.gather

Uses httpx.AsyncClient if available, falls back to urllib (stdlib).
Unhealthy agents are automatically skipped.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .agent import AgentRegistry, AgentStatus, RemoteAgent
from .router import RoutingResult, ToolAssignment
from .tracing import RoutingTrace, TraceStep

if TYPE_CHECKING:
    pass


class ExecutionResult:
    """Result of distributed execution."""

    __slots__ = ('outputs', 'trace', 'success')

    def __init__(self, outputs: List[Dict[str, Any]], trace: RoutingTrace,
                 success: bool) -> None:
        self.outputs = outputs
        self.trace = trace
        self.success = success

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'outputs': self.outputs,
            'trace': self.trace.to_dict(),
        }


async def execute(
    routing_result: RoutingResult,
    registry: AgentRegistry,
    trace: Optional[RoutingTrace] = None,
    client: Any = None,
    timeout: float = 30.0,
) -> ExecutionResult:
    """Execute a routing result against remote agents.

    Args:
        routing_result: The RoutingResult from route().
        registry: AgentRegistry with remote agent info.
        trace: Optional RoutingTrace to record execution steps.
        client: Optional httpx.AsyncClient (reuse for connection pooling).
        timeout: HTTP request timeout in seconds.

    Returns:
        ExecutionResult with outputs and trace.
    """
    if trace is None:
        trace = RoutingTrace(query=routing_result.metadata.get('query', ''))

    trace.start_execution()

    assignments = routing_result.tool_assignments
    if not assignments:
        trace.finish_execution()
        return ExecutionResult(outputs=[], trace=trace, success=True)

    # Determine execution mode from decomposition
    mode = _detect_mode(routing_result)

    if mode == 'single':
        outputs = await _execute_single(
            assignments[0], registry, trace, client, timeout)
    elif mode == 'parallel':
        outputs = await _execute_parallel(
            assignments, registry, trace, client, timeout)
    else:  # sequential
        outputs = await _execute_sequential(
            assignments, registry, trace, client, timeout)

    trace.finish_execution()
    success = all(s.status == 'success' for s in trace.steps
                  if s.status != 'skipped')
    return ExecutionResult(outputs=outputs, trace=trace, success=success)


def _detect_mode(routing_result: RoutingResult) -> str:
    """Detect execution mode from the routing result."""
    if len(routing_result.tool_assignments) <= 1:
        return 'single'

    decomp = routing_result.decomposition
    if decomp and hasattr(decomp, 'mode'):
        return decomp.mode  # 'sequential', 'parallel', or 'single'

    strategy = routing_result.strategy
    if 'parallel' in strategy:
        return 'parallel'
    return 'sequential'


async def _execute_single(
    assignment: ToolAssignment,
    registry: AgentRegistry,
    trace: RoutingTrace,
    client: Any,
    timeout: float,
) -> List[Dict[str, Any]]:
    """Execute a single tool assignment."""
    agent = registry.agent_for_tool(assignment.tool)
    step = TraceStep(
        agent_id=agent.id if agent else '',
        tool_name=assignment.tool,
        subtask=assignment.subtask,
    )
    trace.add_step(step)

    if not agent or not agent.is_healthy:
        step.skip('agent unhealthy' if agent else 'agent not found')
        return [{'tool': assignment.tool, 'error': step.error}]

    result = await _call_agent(agent, assignment, client, timeout, step)
    return [result]


async def _execute_sequential(
    assignments: List[ToolAssignment],
    registry: AgentRegistry,
    trace: RoutingTrace,
    client: Any,
    timeout: float,
) -> List[Dict[str, Any]]:
    """Execute assignments sequentially, piping context forward."""
    outputs: List[Dict[str, Any]] = []
    context: Optional[Dict[str, Any]] = None

    for assignment in assignments:
        agent = registry.agent_for_tool(assignment.tool)
        step = TraceStep(
            agent_id=agent.id if agent else '',
            tool_name=assignment.tool,
            subtask=assignment.subtask,
        )
        trace.add_step(step)

        if not agent or not agent.is_healthy:
            step.skip('agent unhealthy' if agent else 'agent not found')
            outputs.append({'tool': assignment.tool, 'error': step.error})
            continue

        result = await _call_agent(
            agent, assignment, client, timeout, step, context=context)
        outputs.append(result)

        # Pass output as context to next step
        if result.get('output'):
            context = result['output']

    return outputs


async def _execute_parallel(
    assignments: List[ToolAssignment],
    registry: AgentRegistry,
    trace: RoutingTrace,
    client: Any,
    timeout: float,
) -> List[Dict[str, Any]]:
    """Execute assignments in parallel with asyncio.gather."""
    steps = []
    tasks = []

    for assignment in assignments:
        agent = registry.agent_for_tool(assignment.tool)
        step = TraceStep(
            agent_id=agent.id if agent else '',
            tool_name=assignment.tool,
            subtask=assignment.subtask,
        )
        trace.add_step(step)
        steps.append(step)

        if not agent or not agent.is_healthy:
            step.skip('agent unhealthy' if agent else 'agent not found')
            tasks.append(_make_skip_result(assignment.tool, step.error))
        else:
            tasks.append(
                _call_agent(agent, assignment, client, timeout, step))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    outputs = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            steps[i].finish(error=str(result))
            outputs.append({'tool': assignments[i].tool, 'error': str(result)})
        else:
            outputs.append(result)

    return outputs


async def _make_skip_result(tool: str, error: str) -> Dict[str, Any]:
    """Return a skip result (async for gather compatibility)."""
    return {'tool': tool, 'error': error}


async def _call_agent(
    agent: RemoteAgent,
    assignment: ToolAssignment,
    client: Any,
    timeout: float,
    step: TraceStep,
    context: Any = None,
) -> Dict[str, Any]:
    """Call a remote agent's execute endpoint."""
    step.start()

    payload = {
        'tool': assignment.tool,
        'subtask': assignment.subtask,
        'parameters': assignment.parameters,
    }
    if context is not None:
        payload['context'] = context

    try:
        response = await _http_post(
            agent.endpoint, payload, client, timeout)
        step.finish(result=response)
        return {
            'tool': assignment.tool,
            'agent_id': agent.id,
            'output': response,
        }
    except Exception as e:
        step.finish(error=str(e))
        return {
            'tool': assignment.tool,
            'agent_id': agent.id,
            'error': str(e),
        }


async def _http_post(
    url: str,
    payload: Dict[str, Any],
    client: Any = None,
    timeout: float = 30.0,
) -> Any:
    """POST JSON to a URL. Uses httpx if client provided or available."""
    if client is not None:
        # Assume httpx.AsyncClient
        resp = await client.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    # Try httpx
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as c:
            resp = await c.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except ImportError:
        pass

    # Fallback to urllib
    return await _http_post_urllib(url, payload, timeout)


async def _http_post_urllib(
    url: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Any:
    """POST JSON using urllib (stdlib fallback)."""
    import urllib.request
    import urllib.error

    def _do_post() -> Any:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do_post)
