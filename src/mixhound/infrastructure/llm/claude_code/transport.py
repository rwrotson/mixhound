# Vendored from h0rv/pydantic-ai-claude-code (MIT, (c) 2026 Armin Ronacher, chem), adapted:
# plain instructions replace Claude Code's preset system prompt, as V1 research did.
import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NoReturn, TypeGuard

import anyio
import httpx2
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookContext,
    HookInput,
    HookJSONOutput,
    HookMatcher,
    McpServerConfig,
    Message,
    RateLimitEvent,
    ResultError,
    ResultMessage,
    SdkMcpTool,
    StreamEvent,
    SystemMessage,
    create_sdk_mcp_server,
)
from claude_agent_sdk import query as sdk_query
from claude_agent_sdk.types import HookEvent
from pydantic import JsonValue, TypeAdapter
from pydantic_ai.exceptions import (
    ContentFilterError,
    ModelRateLimitError,
    ModelTimeoutError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.tools import ToolDefinition

from mixhound.infrastructure.llm.claude_code.auth import subscription_account
from mixhound.infrastructure.llm.claude_code.errors import (
    ClaudeCodeCreditLimitError,
    ClaudeCodeQuotaError,
    ClaudeCodeSubscriptionLimitError,
    ClaudeCodeUsageCreditsError,
)
from mixhound.infrastructure.llm.claude_code.request import (
    CLI_HEADERS,
    application_headers,
    extra_body,
    input_frames,
    object_value,
    prepare,
    system_prompt,
    text_value,
)

_OBJECT = TypeAdapter(dict[str, JsonValue])
_OBJECTS = TypeAdapter(list[dict[str, JsonValue]])
_LIMIT = 64 * 1024 * 1024
_MAX_REQUEST_BYTES = 10 * 1024 * 1024
_TIMEOUTS = TypeAdapter(dict[str, float | None])
_TOO_MANY_REQUESTS = 429
_SERVER = "pydantic_ai"

type Hooks = dict[HookEvent, list[HookMatcher]]


def _client(options: ClaudeAgentOptions) -> ClaudeSDKClient:
    """Construct the public SDK client; offline tests replace this function."""
    return ClaudeSDKClient(options=options)


def _query(
    prompt: AsyncIterable[dict[str, JsonValue]], options: ClaudeAgentOptions
) -> AsyncIterator[Message]:
    """Start one SDK query; the reader task consumes it to EOF, trailing errors included."""
    return sdk_query(prompt=prompt, options=options)


def _exception_group(error: Exception) -> TypeGuard[ExceptionGroup[Exception]]:
    return isinstance(error, ExceptionGroup)


def _failure(error: Exception) -> Exception:
    """Unwrap single-member exception groups, keeping their notes."""
    if _exception_group(error) and len(error.exceptions) == 1:
        failure = _failure(error.exceptions[0])
        for note in getattr(error, "__notes__", []):
            failure.add_note(note)
        return failure
    return error


def default_options() -> ClaudeAgentOptions:
    """Options of a bare model call: no Claude Code tools, settings, skills or sessions."""
    return ClaudeAgentOptions(
        tools=[],
        max_turns=1,
        setting_sources=[],
        skills=[],
        strict_mcp_config=True,
        extra_args={"no-session-persistence": None},
    )


def _sdk_options(options: ClaudeAgentOptions | None) -> ClaudeAgentOptions:
    # Reuse the SDK dataclass so new SDK fields pass through automatically.
    if options is None:
        return default_options()
    return replace(options, env=dict(options.env), extra_args=dict(options.extra_args))


async def inspect_cli(sdk_options: ClaudeAgentOptions | None = None) -> dict[str, JsonValue]:
    """Read public SDK initialization information without sending a prompt."""
    stderr: list[str] = []
    supplied = _sdk_options(sdk_options)

    def on_stderr(line: str) -> None:
        stderr.append(line)
        if supplied.stderr is not None:
            supplied.stderr(line)

    try:
        async with _client(replace(supplied, stderr=on_stderr)) as client:
            info = await client.get_server_info()
    except Exception as error:
        for line in stderr:
            error.add_note("Claude SDK stderr: " + line)
        raise
    return _OBJECT.validate_python(info or {}, strict=True)


@dataclass(frozen=True)
class Limits:
    """How long one SDK call may stay silent or run, and whether paid credits are accepted."""

    idle_timeout: float
    total_timeout: float
    allow_usage_credits: bool


class ClaudeCodeTransport(httpx2.AsyncBaseTransport):
    """Adapt mapped Anthropic requests to the SDK without network fallback."""

    def __init__(self, sdk_options: ClaudeAgentOptions | None, limits: Limits) -> None:
        self.sdk_options = _sdk_options(sdk_options)
        self.limits = limits
        self._account: dict[str, JsonValue] | None = None
        self._auth_lock = asyncio.Lock()

    async def verify_subscription(self) -> None:
        """Check once that the CLI is logged in with an eligible subscription."""
        async with self._auth_lock:
            if self._account is not None:
                return
            info = await inspect_cli(self.sdk_options)
            self._account = subscription_account(info)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Answer a streaming POST /v1/messages from one SDK query."""
        if request.method != "POST" or request.url.path != "/v1/messages" or request.url.query:
            raise UserError("The SDK transport supports streaming POST /v1/messages only.")
        raw = await request.aread()
        if len(raw) > _MAX_REQUEST_BYTES:
            raise UserError("SDK request exceeds the 10 MB limit.")
        body = _OBJECT.validate_json(raw, strict=True)
        if body.get("stream") is not True:
            raise UserError("The SDK transport requires native streaming requests.")
        messages = _OBJECTS.validate_python(body.get("messages"), strict=True)
        if any(message.get("role") == "system" for message in messages):
            raise UserError(
                "The SDK transport cannot preserve inline system history; it will not flatten it."
            )
        desired, tools, names = prepare(body)
        values = _TIMEOUTS.validate_python(request.extensions.get("timeout", {}), strict=True)
        headers = application_headers(
            (name, value)
            for name, value in request.headers.multi_items()
            if name not in CLI_HEADERS
            and name != "anthropic-beta"
            and not name.startswith("x-stainless-")
        )
        call = _Call(
            desired=desired,
            catalog=tools,
            names=names,
            beta=request.headers.get("anthropic-beta", ""),
            request_timeout=values.get("read"),
            headers=headers,
        )
        # This is an in-memory adapter response, NOT an upstream HTTP status.
        # The Agent SDK does not expose the upstream status or response headers.
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=_SDKStream(self, call)
        )


@dataclass(frozen=True)
class _Call:
    """One native request as the SDK stream needs it."""

    desired: dict[str, JsonValue]
    catalog: list[ToolDefinition]
    names: dict[str, str]
    beta: str
    request_timeout: float | None
    headers: list[tuple[str, str]]


class _SDKStream(httpx2.AsyncByteStream):
    def __init__(self, transport: ClaudeCodeTransport, call: _Call) -> None:
        self.transport = transport
        self.call = call
        self.desired = call.desired
        self.names = call.names
        self.model = text_value(call.desired.get("model"))
        self.pending: list[bytes] = []
        self.buffered = 0
        self.hold = False
        self.complete = False
        self.started = False
        self.has_tools = False
        self.refused = False
        self.stop_reason: str | None = None
        self.result: ResultMessage | None = None
        self.limit_event: RateLimitEvent | None = None
        self.credit_event: RateLimitEvent | None = None
        self.native_error: dict[str, JsonValue] | None = None
        self.replay_waiting = False
        self.replay_failed = False
        self.replay_acknowledged = asyncio.Event()
        self.invoked: list[dict[str, JsonValue]] = []
        self.stderr: list[str] = []
        self._queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue(maxsize=16)
        self._worker: asyncio.Task[None] | None = None
        self._cancel_sent = False
        self._closed = False
        self._reader_scope: anyio.CancelScope | None = None
        self._reader_stopping = False
        # Validate unsupported settings before any SDK client or preflight starts.
        self.options = self._options()

    async def _inert_tool(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        self.invoked.append(arguments)
        return {
            "isError": True,
            "content": [
                {"type": "text", "text": "Tools execute only in the Pydantic AI agent loop."}
            ],
        }

    async def _defer_tool(
        self, data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        del tool_use_id, context  # The SDK hook signature; only the input decides.
        if data["hook_event_name"] != "PreToolUse" or data["tool_name"] not in self.names:
            return {}
        return {
            "hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "defer"}
        }

    def _native_extra(self) -> dict[str, JsonValue]:
        """API options for CLAUDE_CODE_EXTRA_BODY, minus what SDK options already set."""
        supplied = self.transport.sdk_options
        extra = extra_body(self.desired)
        # MCP supplies tool schemas through the SDK itself, including any
        # caller-configured SDK tools. Never replace its complete tool catalog.
        native_tools = _OBJECTS.validate_python(extra.pop("tools", []), strict=True)
        if any(tool.get("strict") is True or "cache_control" in tool for tool in native_tools):
            raise UserError(
                "SDK MCP tools do not expose API tool strict/cache-control overrides; "
                "use Pydantic validation and SDK caching."
            )
        if supplied.thinking is not None or supplied.max_thinking_tokens is not None:
            extra.pop("thinking", None)
            if "context_management" not in self.desired:
                extra.pop("context_management", None)
        output = object_value(extra.get("output_config", {}))
        for name, preset in (
            ("effort", supplied.effort is not None),
            ("format", supplied.output_format is not None),
            ("task_budget", supplied.task_budget is not None),
        ):
            if preset:
                output.pop(name, None)
        if output:
            extra["output_config"] = output
        else:
            extra.pop("output_config", None)
        return extra

    def _env(self) -> dict[str, str]:
        supplied = self.transport.sdk_options
        configured = _OBJECT.validate_json(
            supplied.env.get("CLAUDE_CODE_EXTRA_BODY", "{}"), strict=True
        )
        # Suppress SDK auxiliary inference by default; explicit SDK env configuration wins.
        env = {
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            **supplied.env,
            "CLAUDE_CODE_EXTRA_BODY": json.dumps({**configured, **self._native_extra()}),
        }
        if self.call.headers:
            custom = "\n".join(name + ": " + value for name, value in self.call.headers)
            env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(
                filter(None, (supplied.env.get("ANTHROPIC_CUSTOM_HEADERS"), custom))
            )
        return env

    def _tool_server(self) -> tuple[dict[str, McpServerConfig] | str | Path, Hooks | None]:
        """Add our tools as an inert MCP server whose calls are deferred to pydantic-ai."""
        supplied = self.transport.sdk_options
        tools = [
            SdkMcpTool[dict[str, JsonValue]](
                tool.name, tool.description or "", tool.parameters_json_schema, self._inert_tool
            )
            for tool in self.call.catalog
        ]
        if not tools:
            return supplied.mcp_servers, supplied.hooks
        servers = supplied.mcp_servers
        if not isinstance(servers, dict):
            raise UserError(
                "Combining SDK MCP config files with Pydantic tools "
                "requires SDK dictionary-form mcp_servers."
            )
        if _SERVER in servers:
            raise UserError("The pydantic_ai MCP server name is reserved for Pydantic tools.")
        hooks: Hooks = {event: [*matchers] for event, matchers in (supplied.hooks or {}).items()}
        hooks.setdefault("PreToolUse", []).append(HookMatcher(hooks=[self._defer_tool]))
        return {**servers, _SERVER: create_sdk_mcp_server(_SERVER, tools=tools)}, hooks

    def _extra_args(self) -> dict[str, str | None]:
        extra_args = dict(self.transport.sdk_options.extra_args)
        if self.call.beta:
            betas = [
                *self.transport.sdk_options.betas,
                *(extra_args.get("betas") or "").split(","),
                *self.call.beta.split(","),
            ]
            extra_args["betas"] = ",".join(dict.fromkeys(betas)).strip(",")
        return extra_args

    def _options(self) -> ClaudeAgentOptions:
        supplied = self.transport.sdk_options
        instructions = system_prompt(self.desired)
        if instructions and supplied.system_prompt is not None:
            raise UserError(
                "Use either SDK system_prompt or Pydantic instructions, "
                "not two conflicting system prompts."
            )
        env = self._env()
        servers, hooks = self._tool_server()

        def on_stderr(line: str) -> None:
            self.stderr.append(line)
            if supplied.stderr is not None:
                supplied.stderr(line)

        # All other SDK fields, including future fields and live callback/store
        # objects, pass through via the SDK dataclass without copying its schema.
        return replace(
            supplied,
            model=self.model,
            tools=supplied.tools if supplied.tools is not None else [],
            system_prompt=supplied.system_prompt
            if supplied.system_prompt is not None
            else instructions,
            mcp_servers=servers,
            hooks=hooks,
            include_partial_messages=True,
            env=env,
            extra_args=self._extra_args(),
            stderr=on_stderr,
        )

    def _result(self, message: ResultMessage) -> None:
        if self.replay_waiting:
            if message.num_turns != 0:
                raise UnexpectedModelBehavior(
                    "SDK generated during zero-turn history replay: " + str(asdict(message))
                )
            self.replay_failed = message.is_error
            self.replay_waiting = False
            self.replay_acknowledged.set()
            if message.is_error:
                self.result = message
            return
        if message.num_turns == 0 or self.result is not None:
            raise UnexpectedModelBehavior(
                "SDK emitted an unexpected result: " + str(asdict(message))
            )
        self.result = message

    def _check_init(self, data: dict[str, JsonValue]) -> None:
        if data.get("apiKeySource") != "none":
            raise UserError("Claude subscription login is required: " + json.dumps(data))

    def _rate_limit(self, message: RateLimitEvent) -> None:
        info = message.rate_limit_info
        # Do not abort here: the following terminal SDK error can carry the
        # actual provider failure, request ID, and additional diagnostics.
        if info.status == "rejected":
            self.limit_event = message
        if info.raw.get("isUsingOverage") is True or info.raw.get("overageInUse") is True:
            self.credit_event = message

    async def _message(self, message: Message) -> None:
        if isinstance(message, StreamEvent):
            if message.parent_tool_use_id is not None:
                raise UnexpectedModelBehavior("SDK attempted subagent activity.")
            if self.replay_waiting:
                raise UnexpectedModelBehavior("SDK generated during zero-turn history replay.")
            await self._event(_OBJECT.validate_python(message.event, strict=True))
        elif isinstance(message, SystemMessage) and message.subtype == "init":
            self._check_init(_OBJECT.validate_python(message.data, strict=True))
        elif isinstance(message, RateLimitEvent):
            self._rate_limit(message)
        elif isinstance(message, ResultMessage):
            self._result(message)

    async def _frames(self) -> AsyncGenerator[dict[str, JsonValue]]:
        for frame in input_frames(self.desired):
            # Pydantic-owned text is data, never a CLI slash command or @file.
            composed: dict[str, JsonValue] = {**frame, "client_composed": True}
            self.replay_waiting = composed.get("shouldQuery") is False
            self.replay_acknowledged.clear()
            yield composed
            if self.replay_waiting:
                await self.replay_acknowledged.wait()
            if self.replay_failed:
                return

    async def _read_query(self, incoming: asyncio.Queue[Message | Exception | None]) -> None:
        # SDK cleanup uses AnyIO shielding. Cancel its scope, never Task.cancel()
        # during cleanup, so repeated caller cancellation cannot interrupt close.
        received = 0
        try:
            with anyio.CancelScope() as scope:
                self._reader_scope = scope
                if self._reader_stopping:
                    scope.cancel()
                await self.transport.verify_subscription()
                async for message in _query(self._frames(), self.options):
                    received += len(json.dumps(asdict(message)).encode())
                    if received > _LIMIT:
                        incoming.put_nowait(
                            UnexpectedModelBehavior("SDK response exceeds the 64 MB limit.")
                        )
                        scope.cancel()
                        continue
                    incoming.put_nowait(message)
        except Exception as exc:  # noqa: BLE001 - native SDK errors propagate to the caller unchanged.
            incoming.put_nowait(exc)
        finally:
            self._reader_scope = None
            incoming.put_nowait(None)

    async def _join_reader(self, reader: asyncio.Task[None]) -> None:
        self._reader_stopping = True
        if self._reader_scope is not None:
            self._reader_scope.cancel()
        await _join(reader)

    async def _next_message(
        self, incoming: asyncio.Queue[Message | Exception | None]
    ) -> Message | Exception | None:
        timeout = self.transport.limits.idle_timeout
        request_read = self.call.request_timeout
        if request_read is not None:
            timeout = min(timeout, request_read)
        try:
            async with asyncio.timeout(timeout):
                return await incoming.get()
        except TimeoutError as exc:
            if request_read is not None and request_read <= self.transport.limits.idle_timeout:
                raise ModelTimeoutError(
                    self.model, "Claude SDK exceeded the request read timeout."
                ) from exc
            raise

    def _quota_error(
        self, *, body: object | None = None
    ) -> ClaudeCodeQuotaError | ModelRateLimitError | None:
        event = self.limit_event
        if event is not None and event.rate_limit_info.rate_limit_type == "overage":
            return ClaudeCodeCreditLimitError(self.model, event, body=body)
        if self.credit_event is not None:
            if not self.transport.limits.allow_usage_credits:
                return ClaudeCodeUsageCreditsError(self.model, self.credit_event, body=body)
            return None
        if event is None:
            return None
        if event.rate_limit_info.rate_limit_type is not None:
            return ClaudeCodeSubscriptionLimitError(self.model, event, body=body)
        return ModelRateLimitError(
            self.model, "Claude SDK rate limit: " + json.dumps(asdict(event)), body=asdict(event)
        )

    def _raise_result_error(self, error: ResultError) -> NoReturn:
        if error.api_error_status == _TOO_MANY_REQUESTS:
            quota = self._quota_error(body=error.data)
            if isinstance(quota, ClaudeCodeQuotaError):
                raise quota from error
        raise error

    async def _run(self) -> None:
        incoming: asyncio.Queue[Message | Exception | None] = asyncio.Queue()
        reader = asyncio.create_task(self._read_query(incoming))
        try:
            while (message := await self._next_message(incoming)) is not None:
                if isinstance(message, ResultError):
                    self._raise_result_error(message)
                if isinstance(message, Exception):
                    raise message
                await self._message(message)
        finally:
            await self._join_reader(reader)
        self._check_outcome()

    def _check_outcome(self) -> None:
        """Raise unless the SDK delivered exactly one complete, permitted model response."""
        if self.result is not None and self.result.is_error:
            # Zero-exit CLI failures may supply only the SDK's typed message.
            self._raise_result_error(
                ResultError(
                    "Claude SDK error: " + json.dumps(asdict(self.result)),
                    data=asdict(self.result),
                )
            )
        if quota := self._quota_error():
            raise quota
        if self.invoked:
            raise UnexpectedModelBehavior(
                "SDK attempted to execute a proposal-only tool instead of deferring it."
            )
        if not self.complete or self.result is None:
            raise UnexpectedModelBehavior("SDK did not complete one native model response.")
        if self.refused:
            raise ContentFilterError("The model refused to complete the request.")

    def _start(self, kind: JsonValue) -> None:
        if self.complete and kind != "ping":
            raise UnexpectedModelBehavior("SDK emitted data after message_stop.")
        if kind == "message_start":
            if self.started:
                raise UnexpectedModelBehavior("SDK attempted more than one model response.")
            self.started = True
        elif kind not in ("ping", "error") and not self.started:
            raise UnexpectedModelBehavior("SDK emitted content before message_start.")

    def _tool_block(self, event: dict[str, JsonValue]) -> None:
        block = object_value(event.get("content_block"))
        if block.get("type") != "tool_use":
            return
        name = text_value(block.get("name"))
        if name not in self.names:
            raise UnexpectedModelBehavior("SDK proposed an undeclared tool.")
        block["name"] = self.names[name]
        event["content_block"] = block
        self.hold = self.has_tools = True

    def _track(self, kind: JsonValue, event: dict[str, JsonValue]) -> None:
        """Note tool proposals, stop reasons and errors; from the first one on, hold output."""
        if kind == "content_block_start":
            self._tool_block(event)
        elif kind == "message_delta":
            reason = object_value(event.get("delta")).get("stop_reason")
            if isinstance(reason, str):
                self.stop_reason = reason
            self.refused = self.stop_reason == "refusal"
            self.hold = True
        elif kind == "error":
            self.native_error = event
            self.hold = True
        elif kind == "message_stop":
            if not self.stop_reason:
                raise UnexpectedModelBehavior("SDK stopped without a complete response.")
            self.complete = self.hold = True

    async def _event(self, event: dict[str, JsonValue]) -> None:
        kind = event.get("type")
        self._start(kind)
        self._track(kind, event)
        raw = f"event: {kind}\ndata: {json.dumps(event)}\n\n".encode()
        if self.hold:
            self.buffered += len(raw)
            if self.buffered > _LIMIT:
                raise UnexpectedModelBehavior("SDK tool response exceeds the 64 MB buffer limit.")
            self.pending.append(raw)
        else:
            await self._queue.put(raw)

    def _annotate(self, failure: Exception) -> None:
        for line in self.stderr:
            failure.add_note("Claude SDK stderr: " + line)
        for label, data in (
            ("rate-limit event", self.limit_event),
            ("usage-credit event", self.credit_event),
            ("native stream error", self.native_error),
        ):
            if data is not None:
                detail = asdict(data) if isinstance(data, RateLimitEvent) else data
                failure.add_note("Claude SDK " + label + ": " + json.dumps(detail))

    async def _produce(self) -> None:
        try:
            async with asyncio.timeout(self.transport.limits.total_timeout):
                await self._run()
            # SDK cleanup finishes before a tool proposal can reach Pydantic AI.
            for chunk in self.pending:
                await self._queue.put(chunk)
            await self._queue.put(None)
        except Exception as exc:  # noqa: BLE001 - every SDK error reaches the caller as raised.
            failure = _failure(exc)
            self._annotate(failure)
            await self._queue.put(failure)

    async def _iterate(self) -> AsyncGenerator[bytes]:
        if self._worker is not None:
            raise RuntimeError("SDK streams can only be consumed once.")
        self._worker = asyncio.create_task(self._produce())
        try:
            while not self._closed:
                item = await self._queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            await self.aclose()

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def aclose(self) -> None:
        """Stop the SDK worker and wait until its cleanup finished."""
        self._closed = True
        worker = self._worker
        if worker is None:
            return
        if not self._cancel_sent:
            self._cancel_sent = True
            worker.cancel()
        await _join(worker)


async def _join(task: asyncio.Task[None]) -> None:
    """Wait for a task through repeated cancellation; re-raise its error unless it was cancelled."""
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    break
        if not task.cancelled():
            task.result()
