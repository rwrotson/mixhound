import pytest
from pydantic_ai.exceptions import (
    ContentFilterError,
    FallbackExceptionGroup,
    ModelAuthenticationError,
    ModelHTTPError,
    ModelNotFoundError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    RunTimedOut,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)

from mixhound.agents.clock import BudgetExceededError
from mixhound.agents.failures import classify_failure, failure_code


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (TimeoutError(), "timeout"),
        (RunTimedOut("deadline"), "timeout"),
        (BudgetExceededError("cap"), "budget_exceeded"),
        (UsageLimitExceeded("requests"), "limit_reached"),
        (ModelAuthenticationError("m", "bad key"), "auth_unavailable"),
        (ModelNotFoundError("m", "unknown"), "model_unavailable"),
        (ModelQuotaExceededError("m", "quota"), "limit_reached"),
        (ModelRateLimitError("m", "slow down"), "limit_reached"),
        (ModelOverloadedError("m", "busy"), "limit_reached"),
        (UnexpectedModelBehavior("bad json"), "invalid_result"),
        (ContentFilterError("refused"), "invalid_result"),
        (ModelHTTPError(403, "m"), "auth_unavailable"),
        (ModelHTTPError(404, "m"), "model_unavailable"),
        (ModelHTTPError(529, "m"), "limit_reached"),
        (ModelHTTPError(500, "m", body="usage limit hit"), "limit_reached"),
        (ModelHTTPError(500, "m"), "process_failure"),
        (RuntimeError("Claude subscription login is required"), "auth_unavailable"),
        (RuntimeError("anything else"), "process_failure"),
        (
            FallbackExceptionGroup("all", [ModelNotFoundError("m", "unknown"), TimeoutError()]),
            "model_unavailable",
        ),
    ],
)
def test_errors_map_to_v1_failure_codes(error: BaseException, code: str) -> None:
    assert failure_code(error) == code


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("Not logged in", "auth_unavailable"),
        ("max_budget reached", "budget_exceeded"),
        ("You've hit your 5-hour limit", "limit_reached"),
        ("model_not_found", "model_unavailable"),
        ("connection reset by peer", "search_unavailable"),
        ("all fine", None),
    ],
)
def test_messages_are_classified_in_v1_order(message: str, code: str | None) -> None:
    assert classify_failure(message) == code
