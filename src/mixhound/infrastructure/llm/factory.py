from collections.abc import Mapping
from typing import Literal

from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings, ThinkingEffort

from mixhound.config import LlmSettings
from mixhound.infrastructure.llm.claude_code import ClaudeCodeModel, ClaudeCodeOptions
from mixhound.infrastructure.llm.claude_code.transport import default_options

CLAUDE_CODE = "claude-code:"
OPENROUTER = "openrouter:"
# Expensive Claude models fall back to this one when overloaded, as in V1.
CLAUDE_FALLBACK_MODEL = "sonnet"
# Variables a parent Claude Code session exports; inheriting them would make the research
# process look like a nested child session or override the requested effort. The SDK
# itself drops CLAUDECODE and sets CLAUDE_CODE_ENTRYPOINT.
CLAUDE_SESSION_ENV = (
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_PID",
    "CLAUDE_EFFORT",
)

type Effort = Literal["low", "medium", "high", "xhigh", "max", "ultra"]


def _thinking(effort: Effort) -> ThinkingEffort:
    """Map a run's reasoning effort onto pydantic-ai's levels; providers clamp further."""
    return "max" if effort == "ultra" else effort


class ModelFactory:
    """Builds and caches the models a run names; closes Claude Code clients on exit."""

    def __init__(self, settings: LlmSettings, environ: Mapping[str, str]) -> None:
        self.llm = settings
        self.environ = environ
        self._models: dict[str, Model] = {}
        self._claude: list[ClaudeCodeModel] = []

    def model(self, name: str | None = None, *, helper: bool = False) -> Model:
        """Return the named model, else the configured (helper) default, with its fallback."""
        name = name or (self.llm.helper_model if helper else None) or self.llm.model
        if name not in self._models:
            primary = self._build(name)
            fallback = self.llm.fallback_model
            self._models[name] = (
                FallbackModel(primary, self._build(fallback))
                if fallback is not None and fallback != name
                else primary
            )
        return self._models[name]

    def settings(self, effort: Effort | None) -> ModelSettings:
        """Return the request settings of a reasoning effort; None keeps the model default."""
        return ModelSettings() if effort is None else ModelSettings(thinking=_thinking(effort))

    def _build(self, name: str) -> Model:
        if name.startswith(CLAUDE_CODE):
            return self._claude_code(name.removeprefix(CLAUDE_CODE))
        if name.startswith(OPENROUTER) and self.llm.openrouter_api_key is not None:
            provider = OpenRouterProvider(api_key=self.llm.openrouter_api_key.get_secret_value())
            return OpenRouterModel(name.removeprefix(OPENROUTER), provider=provider)
        return infer_model(name)

    def _claude_code(self, name: str) -> ClaudeCodeModel:
        sdk = default_options()
        sdk.cli_path = self.llm.claude_cli_path
        sdk.env = {variable: "" for variable in CLAUDE_SESSION_ENV if variable in self.environ}
        lowered = name.casefold()
        if "opus" in lowered or lowered == "fable":
            sdk.fallback_model = CLAUDE_FALLBACK_MODEL
        model = ClaudeCodeModel(
            name,
            ClaudeCodeOptions(
                sdk,
                idle_timeout=self.llm.claude_idle_timeout,
                allow_usage_credits=self.llm.claude_allow_usage_credits,
            ),
        )
        self._claude.append(model)
        return model

    async def aclose(self) -> None:
        """Close the Claude Code clients created so far."""
        for model in self._claude:
            await model.client.close()
        self._claude.clear()
        self._models.clear()
