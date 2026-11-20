from pydantic_ai.exceptions import (
    FallbackExceptionGroup,
    ModelAuthenticationError,
    ModelHTTPError,
    ModelNotFoundError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    OutputTokenLimitExceeded,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)

from mixhound.agents.clock import BudgetExceededError

# Failure codes by words in an error message, in V1's order (research/classification.py).
_FAILURE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "auth_unavailable",
        (
            "not logged in",
            "login",
            "authentication",
            "unauthorized",
            "invalid api key",
            "oauth token",
        ),
    ),
    ("budget_exceeded", ("max_budget", "budget exceeded", "maximum budget")),
    (
        "limit_reached",
        (
            "rate limit",
            "usage limit",
            "quota",
            "credit balance",
            "overloaded",
            "hit your",
            "5-hour limit",
            "weekly limit",
        ),
    ),
    (
        "model_unavailable",
        ("model is not available", "model not found", "unsupported model", "model_not_found"),
    ),
    (
        "search_unavailable",
        (
            "web_search",
            "network error",
            "connection refused",
            "connection reset",
            "failed to connect",
        ),
    ),
)
_UNAUTHORIZED = {401, 403}
_NOT_FOUND = 404
_LIMITED = {429, 529}
_TYPED: tuple[tuple[type[Exception] | tuple[type[Exception], ...], str], ...] = (
    (TimeoutError, "timeout"),
    (BudgetExceededError, "budget_exceeded"),
    (UsageLimitExceeded, "limit_reached"),
    (ModelAuthenticationError, "auth_unavailable"),
    (ModelNotFoundError, "model_unavailable"),
    (
        (ModelQuotaExceededError, ModelRateLimitError, ModelOverloadedError),
        "limit_reached",
    ),
    # Before UnexpectedModelBehavior, its base class.
    (OutputTokenLimitExceeded, "output_limit"),
    (UnexpectedModelBehavior, "invalid_result"),
)


def classify_failure(output: str) -> str | None:
    """Return the failure code an error message reports, if any."""
    message = output.casefold()
    for code, needles in _FAILURE_PATTERNS:
        if any(needle in message for needle in needles):
            return code
    return None


def _http_code(status: int) -> str | None:
    if status in _UNAUTHORIZED:
        return "auth_unavailable"
    if status == _NOT_FOUND:
        return "model_unavailable"
    if status in _LIMITED:
        return "limit_reached"
    return None


def failure_code(error: BaseException) -> str:
    """Name why an agent call failed, with V1's failure codes."""
    if isinstance(error, FallbackExceptionGroup) and error.exceptions:
        return failure_code(error.exceptions[0])
    for kinds, typed in _TYPED:
        if isinstance(error, kinds):
            return typed
    if isinstance(error, ModelHTTPError) and (status := _http_code(error.status_code)):
        return status
    return classify_failure(str(error)) or "process_failure"
