"""OpenTelemetry setup shared by every service.

Spans are always recorded so each request has a trace id. They are exported only when the
standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` (or ``..._TRACES_ENDPOINT``) variable is set, which
covers Langfuse, Jaeger and any other OTLP/HTTP collector. Headers such as Langfuse's basic
auth go in ``OTEL_EXPORTER_OTLP_HEADERS``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_configured = False


def setup_telemetry(service_name: str) -> None:
    global _configured
    if _configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def current_trace_id() -> str:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else ""


class Stopwatch:
    """Millisecond timer used to fill ``timings_ms`` in responses."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 1)


@contextmanager
def timed_span(tracer: trace.Tracer, name: str, **attributes: str | int | float | bool) -> Iterator[trace.Span]:
    """Start a span and record its duration as ``duration_ms`` when it closes."""
    watch = Stopwatch()
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        try:
            yield span
        finally:
            span.set_attribute("duration_ms", watch.ms())
