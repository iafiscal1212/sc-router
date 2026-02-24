"""Tests for RoutingTrace, TraceStep, and TracingHook."""

import time
import pytest
from sc_router.tracing import RoutingTrace, TraceStep, TracingHook


class TestTraceStep:
    def test_start_finish_success(self):
        step = TraceStep(agent_id="a1", tool_name="search", subtask="find X")
        assert step.status == 'pending'

        step.start()
        assert step.status == 'running'
        assert step.started_at > 0

        step.finish(result={"data": "ok"})
        assert step.status == 'success'
        assert step.result == {"data": "ok"}
        assert step.error is None
        assert step.duration_ms >= 0

    def test_start_finish_error(self):
        step = TraceStep(agent_id="a1", tool_name="search", subtask="find X")
        step.start()
        step.finish(error="connection refused")
        assert step.status == 'error'
        assert step.error == "connection refused"
        assert step.result is None

    def test_skip(self):
        step = TraceStep(agent_id="a1", tool_name="search", subtask="find X")
        step.skip("agent unhealthy")
        assert step.status == 'skipped'
        assert step.error == "agent unhealthy"


class TestRoutingTrace:
    def test_classification_timing(self):
        trace = RoutingTrace(query="test query")
        trace.start_classification()
        trace.finish_classification(sc_level=1, strategy='pipeline_sequential',
                                    confidence='high')
        assert trace.sc_level == 1
        assert trace.strategy == 'pipeline_sequential'
        assert trace.confidence == 'high'
        assert trace.classification_ms >= 0

    def test_execution_timing(self):
        trace = RoutingTrace(query="test")
        trace.start_classification()
        trace.finish_classification(0, 'direct')
        trace.start_execution()
        trace.finish_execution()
        assert trace.execution_ms >= 0
        assert trace.total_ms >= 0

    def test_add_step(self):
        trace = RoutingTrace(query="test")
        step = TraceStep(agent_id="a1", tool_name="t1", subtask="s1")
        trace.add_step(step)
        assert len(trace.steps) == 1
        assert trace.steps[0] is step

    def test_success_all_steps_pass(self):
        trace = RoutingTrace(query="test")
        s1 = TraceStep(agent_id="a1", tool_name="t1", subtask="s1")
        s1.start()
        s1.finish(result="ok")
        trace.add_step(s1)

        s2 = TraceStep(agent_id="a2", tool_name="t2", subtask="s2")
        s2.start()
        s2.finish(result="ok")
        trace.add_step(s2)

        assert trace.success is True

    def test_success_with_skipped(self):
        trace = RoutingTrace(query="test")
        s1 = TraceStep(agent_id="a1", tool_name="t1", subtask="s1")
        s1.start()
        s1.finish(result="ok")
        trace.add_step(s1)

        s2 = TraceStep(agent_id="a2", tool_name="t2", subtask="s2")
        s2.skip("unhealthy")
        trace.add_step(s2)

        assert trace.success is True  # skipped steps don't count

    def test_failure_on_error_step(self):
        trace = RoutingTrace(query="test")
        s1 = TraceStep(agent_id="a1", tool_name="t1", subtask="s1")
        s1.start()
        s1.finish(error="timeout")
        trace.add_step(s1)

        assert trace.success is False

    def test_to_dict(self):
        trace = RoutingTrace(query="hello")
        trace.sc_level = 0
        trace.strategy = 'direct'
        trace.confidence = 'high'
        trace.classification_ms = 5.0
        trace.execution_ms = 10.0
        trace.total_ms = 15.0

        d = trace.to_dict()
        assert d['query'] == 'hello'
        assert d['sc_level'] == 0
        assert d['strategy'] == 'direct'
        assert d['classification_ms'] == 5.0
        assert isinstance(d['steps'], list)
        assert isinstance(d['metadata'], dict)

    def test_to_kore_mind_trace(self):
        trace = RoutingTrace(query="hello")
        trace.sc_level = 1
        trace.strategy = 'pipeline_sequential'
        trace.classification_ms = 3.0
        trace.execution_ms = 20.0
        trace.total_ms = 23.0

        km = trace.to_kore_mind_trace()
        assert km['type'] == 'sc_routing'
        assert km['query'] == 'hello'
        assert km['timing']['classification_ms'] == 3.0
        assert km['timing']['total_ms'] == 23.0


class TestTracingHook:
    def test_record_and_retrieve(self):
        hook = TracingHook()
        trace = RoutingTrace(query="test")
        hook.record(trace)
        assert hook.count == 1
        assert hook.traces[0] is trace

    def test_listener_called(self):
        hook = TracingHook()
        received = []
        hook.add_listener(lambda t: received.append(t))

        trace = RoutingTrace(query="test")
        hook.record(trace)
        assert len(received) == 1
        assert received[0] is trace

    def test_multiple_listeners(self):
        hook = TracingHook()
        calls_a = []
        calls_b = []
        hook.add_listener(lambda t: calls_a.append(t))
        hook.add_listener(lambda t: calls_b.append(t))

        hook.record(RoutingTrace(query="q1"))
        assert len(calls_a) == 1
        assert len(calls_b) == 1

    def test_remove_listener(self):
        hook = TracingHook()
        calls = []
        listener = lambda t: calls.append(t)
        hook.add_listener(listener)
        hook.record(RoutingTrace(query="q1"))
        assert len(calls) == 1

        hook.remove_listener(listener)
        hook.record(RoutingTrace(query="q2"))
        assert len(calls) == 1  # not called again

    def test_remove_nonexistent_listener(self):
        hook = TracingHook()
        hook.remove_listener(lambda t: None)  # should not raise

    def test_clear(self):
        hook = TracingHook()
        hook.record(RoutingTrace(query="q1"))
        hook.record(RoutingTrace(query="q2"))
        assert hook.count == 2
        hook.clear()
        assert hook.count == 0

    def test_max_traces_limit(self):
        hook = TracingHook()
        hook._max_traces = 5
        for i in range(10):
            hook.record(RoutingTrace(query=f"q{i}"))
        assert hook.count == 5
        assert hook.traces[0].query == "q5"
