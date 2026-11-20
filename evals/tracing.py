from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from pydantic_ai import Agent, InstrumentationSettings


def enable() -> None:
    """Record agent runs as spans so evaluators can read tool calls and results."""
    # pydantic-evals collects a case's spans from the global provider; set it only once.
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())
    Agent.instrument_all(InstrumentationSettings(include_content=True))


def disable() -> None:
    """Stop instrumenting agent runs."""
    Agent.instrument_all(instrument=False)
