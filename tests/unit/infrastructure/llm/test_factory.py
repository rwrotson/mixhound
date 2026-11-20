from pathlib import Path
from typing import Literal

import pytest
from pydantic import SecretStr
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.models.test import TestModel

from mixhound.config import LlmSettings
from mixhound.infrastructure.llm.claude_code import ClaudeCodeModel
from mixhound.infrastructure.llm.factory import ModelFactory


def factory(environ: dict[str, str] | None = None, **settings: object) -> ModelFactory:
    return ModelFactory(LlmSettings.model_validate(settings), environ or {})


async def test_default_model_is_claude_code_and_cached() -> None:
    models = factory()
    model = models.model()
    assert isinstance(model, ClaudeCodeModel)
    assert model.model_name == "sonnet"
    assert models.model("claude-code:sonnet") is model
    await models.aclose()


async def test_claude_code_options_come_from_settings() -> None:
    models = factory(
        {"CLAUDE_CODE_SESSION_ID": "parent", "HOME": "/home"},
        claude_cli_path="/opt/claude",
        claude_allow_usage_credits=True,
    )
    model = models.model("claude-code:opus")
    assert isinstance(model, ClaudeCodeModel)
    transport = model.transport
    sdk = transport.sdk_options
    assert sdk.cli_path == Path("/opt/claude")
    assert sdk.env == {"CLAUDE_CODE_SESSION_ID": ""}
    assert sdk.fallback_model == "sonnet"
    assert sdk.tools == []
    assert transport.limits.allow_usage_credits
    await models.aclose()


async def test_sonnet_has_no_sdk_fallback() -> None:
    models = factory()
    model = models.model("claude-code:sonnet")
    assert isinstance(model, ClaudeCodeModel)
    assert model.transport.sdk_options.fallback_model is None
    await models.aclose()


def test_helper_default_prefers_helper_model() -> None:
    models = factory(model="test", helper_model="openrouter:openai/cheap", openrouter_api_key="key")
    assert isinstance(models.model(helper=True), OpenRouterModel)
    assert isinstance(models.model(), TestModel)
    assert isinstance(models.model("test", helper=True), TestModel)


def test_other_names_go_through_pydantic_ai() -> None:
    assert isinstance(factory().model("test"), TestModel)


def test_fallback_wraps_every_other_model() -> None:
    models = factory(
        model="test",
        fallback_model="openrouter:openai/backup",
        openrouter_api_key=SecretStr("key"),
    )
    model = models.model()
    assert isinstance(model, FallbackModel)
    assert isinstance(model.models[1], OpenRouterModel)
    assert isinstance(models.model("openrouter:openai/backup"), OpenRouterModel)


@pytest.mark.parametrize(
    ("effort", "thinking"),
    [("low", "low"), ("xhigh", "xhigh"), ("max", "max"), ("ultra", "max")],
)
def test_effort_maps_to_thinking(
    effort: Literal["low", "xhigh", "max", "ultra"], thinking: str
) -> None:
    assert factory(max_tokens=None).settings(effort) == {"thinking": thinking}


def test_no_effort_keeps_model_default() -> None:
    assert factory(max_tokens=None).settings(None) == {}


def test_output_ceiling_applies_to_api_models_only() -> None:
    models = factory(model="openrouter:z-ai/glm", helper_model="claude-code:haiku", max_tokens=4096)
    assert models.settings("high") == {"thinking": "high", "max_tokens": 4096}
    assert models.settings(None, "openai:gpt") == {"max_tokens": 4096}
    assert models.settings("high", "claude-code:sonnet") == {"thinking": "high"}
    assert models.settings(None, helper=True) == {}


def test_output_ceiling_is_on_by_default() -> None:
    assert factory(model="test").settings(None) == {"max_tokens": 32768}
