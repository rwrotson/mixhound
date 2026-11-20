from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "MIXHOUND_"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LogSettings(BaseModel):
    """Console and optional rotating file logging."""

    level: LogLevel = Field(default="INFO", description="Root log level")
    console_level: LogLevel = Field(default="DEBUG", description="Minimum level printed to stderr")
    file_level: LogLevel = Field(default="INFO", description="Minimum level written to the file")
    format: Literal["console", "json"] = Field(
        default="console", description="Console log renderer; file logs are always JSON"
    )
    dir: Path | None = Field(
        default=None, description="Log directory; file logging is disabled if unset"
    )
    file_name: str = Field(
        default="mixhound.log", description="Log file basename without directory separators"
    )
    file_max_bytes: int = Field(
        default=4 * 1024 * 1024, ge=0, description="File size that triggers rotation"
    )
    file_backup_count: int = Field(default=5, ge=0, description="Rotated files to keep")

    @field_validator("file_name")
    @classmethod
    def _require_basename(cls, value: str) -> str:
        """Keep the log file inside the configured directory."""
        if value in {"", ".", ".."} or "/" in value or "\\" in value:
            raise ValueError("Log file name must be a basename")
        return value


class ApiSettings(BaseModel):
    """HTTP interface options."""

    docs_enabled: bool = Field(default=True, description="Serve /docs and /openapi.json")
    cors_origins: list[str] = Field(
        default_factory=list, description="Origins allowed to call the API from a browser"
    )


class HttpSettings(BaseModel):
    """Outgoing HTTP requests for public pages and audio previews."""

    timeout: float = Field(default=5.0, gt=0, description="Seconds per request attempt")
    user_agent: str = Field(default="mixhound/0.1", description="User-Agent of page fetches")
    max_retries: int = Field(
        default=2, ge=0, description="Retries of an unreachable host or a transient status"
    )
    backoff_base: float = Field(default=0.5, ge=0, description="First retry delay in seconds")
    backoff_cap: float = Field(default=4.0, ge=0, description="Longest backoff delay in seconds")
    retry_after_cap: float = Field(
        default=5.0, ge=0, description="Longest honoured Retry-After delay in seconds"
    )


class DeadwaxSettings(BaseModel):
    """Track lookups through deadwax; its cache lives under the data directory."""

    contact: str = Field(
        default="https://github.com/rwrotson/mixhound",
        description="Contact URL or address sent to catalogs in the User-Agent",
    )
    adapter_timeout: float = Field(
        default=30.0, gt=0, description="Seconds one deadwax source may take"
    )
    sources: list[str] | None = Field(
        default=None, description="Sources deadwax may ask; unset uses deadwax defaults"
    )
    cache_enabled: bool = Field(default=True, description="Cache deadwax responses on disk")
    lastfm_api_key: SecretStr | None = Field(default=None, description="Last.fm API key")
    discogs_token: SecretStr | None = Field(
        default=None, description="Discogs personal token; needed for Discogs search"
    )
    youtube_api_key: SecretStr | None = Field(default=None, description="YouTube Data API key")
    spotify_client_id: SecretStr | None = Field(default=None, description="Spotify client ID")
    spotify_client_secret: SecretStr | None = Field(
        default=None, description="Spotify client secret"
    )


class SearchSettings(BaseModel):
    """Web search for the research agents."""

    brave_api_key: SecretStr | None = Field(
        default=None, description="Brave Search API key; web search is off without it"
    )
    results: int = Field(default=10, ge=1, le=20, description="Results per web search")
    timeout: float = Field(default=10.0, gt=0, description="Seconds per web search")


class LlmSettings(BaseModel):
    """Language models of the research agents; a run may name its own model."""

    model: str = Field(
        default="claude-code:sonnet",
        description="Default model: claude-code:<m>, openai-codex:<m> or a pydantic-ai name",
    )
    helper_model: str | None = Field(
        default=None, description="Default model of helper calls; unset uses the default model"
    )
    fallback_model: str | None = Field(
        default=None, description="Model tried when the chosen one fails, e.g. openrouter:<model>"
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None, description="OpenRouter API key for openrouter: models"
    )
    claude_cli_path: Path | None = Field(
        default=None, description="Claude Code executable; unset uses the SDK's bundled CLI"
    )
    claude_idle_timeout: float = Field(
        default=300.0, gt=0, description="Seconds a Claude Code response may stay silent"
    )
    claude_allow_usage_credits: bool = Field(
        default=False, description="Accept Claude Code requests billed to paid usage credits"
    )


class StorageSettings(BaseModel):
    """Where sessions, runs and diagnostics are kept."""

    backend: Literal["memory", "files"] = Field(
        default="files", description="memory keeps nothing across restarts; files uses data_dir"
    )


class QueueSettings(BaseModel):
    """The in-process run queue."""

    max_pending: int = Field(
        default=20, ge=1, description="Runs that may wait behind the active one"
    )


class CacheSettings(BaseModel):
    """The lookup cache of page checks and audio features."""

    enabled: bool = Field(default=True, description="Cache page checks and audio features on disk")


class Settings(BaseSettings):
    """Load application settings from MIXHOUND_* variables and .env."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_nested_delimiter="__",
        env_ignore_empty=True,
        extra="ignore",
    )

    data_dir: Path = Field(
        default=Path("data"), description="Directory of stored runs, caches and diagnostics"
    )
    skills_dir: Path = Field(
        default=Path(".agents/skills"), description="Directory of agent skills"
    )
    log: LogSettings = Field(default_factory=LogSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    http: HttpSettings = Field(default_factory=HttpSettings)
    deadwax: DeadwaxSettings = Field(default_factory=DeadwaxSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)


def nested_model(field: FieldInfo) -> type[BaseModel] | None:
    """Return the settings model of a nested section, if the field is one."""
    for candidate in getattr(field.annotation, "__args__", (field.annotation,)):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def env_fields(
    model: type[BaseModel] = Settings, prefix: str = ENV_PREFIX
) -> Iterator[tuple[str, FieldInfo]]:
    """Yield every leaf setting with its environment variable name."""
    for name, field in model.model_fields.items():
        variable = f"{prefix}{name.upper()}"
        section = nested_model(field)
        if section is None:
            yield variable, field
        else:
            yield from env_fields(section, f"{variable}__")


def load_settings() -> Settings:
    """Load settings from the environment and .env."""
    return Settings()
