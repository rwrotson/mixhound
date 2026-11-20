# Vendored from h0rv/pydantic-ai-claude-code (MIT, (c) 2026 Armin Ronacher, chem), adapted.
import math
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import httpx2
from anthropic import APIConnectionError, AsyncAnthropic
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKError
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelConnectionError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import (
    ModelRequestParameters,
    StreamedResponse,
    check_allow_model_requests,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.profiles import ModelProfileSpec, ToolAdditionMode, ToolDeferralMode
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.settings import ModelSettingName, ModelSettings

from mixhound.infrastructure.llm.claude_code.request import application_headers
from mixhound.infrastructure.llm.claude_code.transport import ClaudeCodeTransport, Limits


@dataclass(frozen=True)
class ClaudeCodeOptions:
    """How the model reaches Claude Code: SDK options, time limits and paid-credit consent."""

    sdk: ClaudeAgentOptions | None = None
    idle_timeout: float = 300
    total_timeout: float = 1800
    # Accepts reported paid usage; it cannot prevent the first charge.
    allow_usage_credits: bool = False


class ClaudeCodeModel(AnthropicModel):
    """Run a subscription-authenticated Claude Code through the public Agent SDK.

    Pydantic AI owns history, tools, structured output, and stream mapping; Claude Code owns
    authentication and model access. Tools are proposed by the model and run only by the agent.
    """

    unsupported_settings: ClassVar[frozenset[ModelSettingName]] = (
        AnthropicModel.unsupported_settings | frozenset[ModelSettingName]({"max_retries"})
    )
    supported_tool_addition_modes: ClassVar[frozenset[ToolAdditionMode]] = frozenset()
    supported_tool_deferral_modes: ClassVar[frozenset[ToolDeferralMode]] = frozenset()

    def __init__(
        self,
        model_name: str | None = None,
        options: ClaudeCodeOptions | None = None,
        *,
        settings: ModelSettings | None = None,
        profile: ModelProfileSpec | None = None,
    ) -> None:
        options = options or ClaudeCodeOptions()
        for timeout in (options.idle_timeout, options.total_timeout):
            if not math.isfinite(timeout) or timeout <= 0:
                raise UserError("Claude SDK timeouts must be finite positive seconds.")
        sdk = options.sdk
        if model_name is not None and sdk is not None and sdk.model not in {None, model_name}:
            raise UserError("model_name and sdk_options.model must agree when both are provided.")
        if model_name is None:
            model_name = sdk.model if sdk is not None and sdk.model is not None else "haiku"
        transport = ClaudeCodeTransport(
            sdk,
            Limits(options.idle_timeout, options.total_timeout, options.allow_usage_credits),
        )
        self.transport = transport
        client = AsyncAnthropic(
            api_key="local-sdk-placeholder",
            base_url="https://claude-agent-sdk.invalid",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=transport, trust_env=False),
        )
        super().__init__(
            model_name,
            provider=AnthropicProvider(anthropic_client=client),
            settings=settings,
            profile=profile,
        )

    @cached_property
    def profile(self) -> AnthropicModelProfile:
        """The Anthropic profile without roles Claude Code's streaming input cannot carry."""
        return {
            **super().profile,
            "supports_inline_system_prompts": False,
            "supports_request_scoped_system_prompts": False,
            "supported_native_tools": frozenset(),
        }

    @property
    def system(self) -> str:
        """The provider name reported on responses."""
        return "claude-code"

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        """No provider-native tools: every tool runs in the agent loop."""
        return frozenset()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make one request by draining the native stream."""
        async with self.request_stream(
            messages, model_settings, model_request_parameters
        ) as response:
            async for _ in response:
                pass
            return response.get()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[object] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """Stream one response; SDK errors surface as raised, not as connection errors."""
        check_allow_model_requests()
        for layer in (self.settings, model_settings):
            if headers := (layer or {}).get("extra_headers"):
                application_headers(headers.items())
        retries = (model_settings or {}).get(
            "max_retries", (self.settings or {}).get("max_retries")
        )
        if retries:
            raise UserError(
                "Automatic transport retries are disabled for subscription-backed requests."
            )
        if model_request_parameters.native_tools:
            raise UserError("Only Pydantic AI function and output tools may execute.")
        try:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as response:
                response.provider_details = {
                    **(response.provider_details or {}),
                    "transport": "claude-agent-sdk",
                    "upstream_http_metadata_available": False,
                }
                yield response
        except ModelConnectionError as exc:
            cause = exc.__cause__
            if isinstance(cause, APIConnectionError) and isinstance(
                cause.__cause__, ClaudeSDKError | UserError | UnexpectedModelBehavior
            ):
                error = cause.__cause__
                # Re-raise the SDK error as the caller's error, keeping its own cause.
                raise error from error.__cause__
            raise
