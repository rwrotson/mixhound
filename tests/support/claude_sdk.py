# Offline Claude Agent SDK protocol fake, ported from h0rv/pydantic-ai-claude-code tests (MIT).
import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    Message,
    ProcessError,
    Transport,
    query,
)

FIXTURES = Path(__file__).parents[1] / "unit" / "infrastructure" / "llm" / "fixtures"
PREFIX = "mcp__pydantic_ai__"
RATE_LIMITED = frozenset(
    {
        "rate-limit",
        "quota",
        "quota-rejected",
        "credit-limit",
        "subscription-status",
        "credit-status",
        "unknown-rate-limit",
    }
)
TERMINAL_ERRORS = frozenset({"errors", "status", "subscription-status", "credit-status"})
TOOL_SCENARIOS = frozenset({"tool", "output-tool", "unknown-tool"})
THOUGHTS: list[dict[str, Any]] = [
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "reasoning"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "signed"},
    },
    {"type": "content_block_stop", "index": 0},
]
type Frame = dict[str, Any]


def frames(name: str = "text") -> list[Frame]:
    path = FIXTURES / f"{name}-stream.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def result_frame(updates: Frame | None = None) -> Frame:
    return {**frames()[-1], **(updates or {})}


class ProtocolTransport(Transport):
    """Raw JSON I/O only. Nothing replaces ClaudeSDKClient or its control machinery."""

    def __init__(self, options: ClaudeAgentOptions, scenario: str, server_info: Frame) -> None:
        self.options, self.scenario, self.server_info = options, scenario, server_info
        self.queue: asyncio.Queue[Frame | Exception | None] = asyncio.Queue()
        self.writes: list[Frame] = []
        self.catalog: list[Frame] = []
        self.initialize: Frame = {}
        self.closed = asyncio.Event()
        self.reader_done = asyncio.Event()
        self.started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.release_close.set()
        self.inventory_ready = asyncio.Event()
        self.deferred: Frame | None = None
        self.terminal_result: Frame | None = None
        self.process_error = ProcessError(
            "synthetic SDK crash", exit_code=17, stderr="fixture stderr"
        )

    def emit(self, frame: Frame | Exception | None) -> None:
        self.queue.put_nowait(frame)

    def request(self, request_id: str, request: Frame) -> None:
        self.emit({"type": "control_request", "request_id": request_id, "request": request})

    def mcp(self, request_id: str, message: Frame) -> None:
        self.request(
            request_id,
            {
                "subtype": "mcp_message",
                "server_name": "pydantic_ai",
                "message": {"jsonrpc": "2.0", **message},
            },
        )

    async def connect(self) -> None:
        pass

    def is_ready(self) -> bool:
        return not self.closed.is_set()

    async def end_input(self) -> None:
        if self.scenario == "hang":
            return
        self.emit(self.process_error if self.scenario in TERMINAL_ERRORS else None)

    async def close(self) -> None:
        self.close_started.set()
        await self.release_close.wait()
        self.closed.set()

    async def read_messages(self) -> AsyncIterator[Frame]:
        try:
            while True:
                frame = await self.queue.get()
                if isinstance(frame, Exception):
                    raise frame
                if frame is None:
                    return
                yield frame
        finally:
            self.reader_done.set()

    async def write(self, data: str) -> None:
        for line in data.splitlines():
            frame = json.loads(line)
            self.writes.append(frame)
            if frame["type"] == "control_request":
                self._initialize(frame)
            elif frame["type"] == "control_response":
                self._control_response(frame["response"])
            elif frame["type"] in ("user", "assistant"):
                await self._turn(frame)
            else:
                pytest.fail(f"Unexpected SDK stdin frame: {frame}")

    def _initialize(self, frame: Frame) -> None:
        assert frame["request"]["subtype"] == "initialize"
        self.initialize = frame["request"]
        response = {
            "subtype": "success",
            "request_id": frame["request_id"],
            "response": self.server_info,
        }
        self.emit({"type": "control_response", "response": response})
        servers = self.options.mcp_servers
        if isinstance(servers, dict) and "pydantic_ai" in servers:
            params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "offline-test", "version": "1"},
            }
            self.mcp("mcp-init", {"id": 0, "method": "initialize", "params": params})
        else:
            self.inventory_ready.set()

    def _control_response(self, response: Frame) -> None:
        assert response["subtype"] == "success", response
        match response["request_id"]:
            case "mcp-init":
                assert "result" in response["response"]["mcp_response"]
                self.mcp("mcp-ready", {"method": "notifications/initialized"})
            case "mcp-ready":
                self.mcp("catalog", {"id": 1, "method": "tools/list", "params": {}})
            case "catalog":
                self.catalog = response["response"]["mcp_response"]["result"]["tools"]
                self.inventory_ready.set()
            case "proposal":
                decision = response["response"]["hookSpecificOutput"]["permissionDecision"]
                assert decision == "defer"
                self.emit(result_frame({"result": "", "deferred_tool_use": self.deferred}))
            case _:
                pytest.fail(f"Unexpected control response: {response}")

    async def _turn(self, frame: Frame) -> None:
        await self.inventory_ready.wait()
        if frame["type"] == "assistant":
            return  # Assistant history never triggers inference or needs a result ack.
        if frame.get("shouldQuery") is False:
            self.emit(result_frame({"num_turns": 0, "result": "", "usage": {}}))
            return
        self.started.set()
        init = frames()[0]
        assert isinstance(self.options.mcp_servers, dict)
        init.update(
            tools=[PREFIX + tool["name"] for tool in self.catalog],
            model=self.options.model,
            mcp_servers=[
                {"name": name, "status": "connected"} for name in self.options.mcp_servers
            ],
        )
        self.emit(init)
        self.generate()

    def generate(self) -> None:
        if self.scenario == "hang":
            return
        if self.scenario == "process":
            self.emit(self.process_error)
            return
        if self.scenario in RATE_LIMITED:
            self._rate_limit()
        if self.scenario in TERMINAL_ERRORS:
            self._terminal_error()
            return
        tool = self.scenario in TOOL_SCENARIOS
        name = PREFIX + ("final_result" if self.scenario == "output-tool" else "return_value")
        if self.scenario == "unknown-tool":
            name = PREFIX + "undeclared_tool"
        if self._stream(name, tool=tool):
            self._finish(name, tool=tool)

    def _finish(self, name: str, *, tool: bool) -> None:
        if tool:
            self._propose(name)
        elif self.scenario == "missing-result":
            self.emit(None)
        else:
            final = result_frame()
            if self.scenario == "native":
                final["structured_output"] = {"value": "MCP_BRIDGE_OK"}
            self.emit(final)

    def _rate_limit(self) -> None:
        rate = next(frame for frame in frames() if frame["type"] == "rate_limit_event")
        overage = self.scenario in ("quota", "quota-rejected")
        rate["rate_limit_info"].update(
            status="allowed" if self.scenario == "quota" else "rejected",
            rateLimitType="overage"
            if self.scenario in ("credit-limit", "credit-status")
            else "seven_day",
            isUsingOverage=overage,
            overageInUse=overage,
            future_limit_field={"diagnostic": "synthetic-sensitive-limit-diagnostic"},
        )
        if self.scenario == "unknown-rate-limit":
            rate["rate_limit_info"]["rateLimitType"] = None
        self.emit(rate)

    def _terminal_error(self) -> None:
        error = result_frame(
            {"is_error": True, "result": "", "errors": ["fixture terminal failure"]}
        )
        error["usage"]["future_usage_field"] = {"diagnostic": "synthetic-sensitive-diagnostic"}
        error["permission_denials"] = [
            {"tool_name": "fixture", "tool_input": {"secret": "synthetic-secret"}}
        ]
        if self.scenario != "errors":
            error.update(errors=["synthetic-unredacted-api-error"], api_error_status=429)
        error["future_result_field"] = {"diagnostic": "synthetic-unredacted-future-result"}
        self.terminal_result = error
        self.emit(error)

    def _stream(self, name: str, *, tool: bool) -> bool:
        """Emit the scenario's stream events; False when the stream was cut short."""
        for original in frames("tool" if tool else "text"):
            if original["type"] != "stream_event":
                continue
            frame = json.loads(json.dumps(original).replace("mcp__probe__return_value", name))
            event = frame["event"]
            # The SDK must not reject unknown native fields.
            event["future_native_field"] = {"preserved": True}
            if self.scenario == "subagent":
                frame["parent_tool_use_id"] = "toolu_forbidden_subagent"
            if self.scenario == "missing-stop" and event["type"] == "message_stop":
                continue
            if self.scenario == "truncated" and event["type"] == "content_block_stop":
                self.emit(None)
                return False
            self._alter(event)
            self.emit(frame)
            self._follow(frame, event["type"])
        return True

    def _alter(self, event: Frame) -> None:
        if self.scenario == "refusal" and event["type"] == "message_delta":
            event["delta"]["stop_reason"] = "refusal"
        if self.scenario == "native" and event["type"] == "content_block_delta":
            text = event["delta"]["text"]
            event["delta"]["text"] = '{"value":"MCP_BRIDGE_OK"}' if text == "PRINT_TRAN" else ""
        if self.scenario == "thinking" and "index" in event:
            event["index"] += 1

    def _follow(self, frame: Frame, kind: str) -> None:
        """Emit what the scenario adds after an event."""
        if self.scenario == "duplicate-response" and kind == "message_start":
            self.emit(frame)
        if self.scenario == "restarted-stream" and kind == "message_stop":
            self.emit(
                next(
                    item
                    for item in frames()
                    if item.get("event", {}).get("type") == "message_start"
                )
            )
        if self.scenario == "future-fields" and kind == "message_start":
            future = {"type": "future_native_event", "opaque": {"preserved": True}}
            self.emit({**frame, "event": future})
        if self.scenario == "thinking" and kind == "message_start":
            for thought in THOUGHTS:
                self.emit({**frame, "event": thought})

    def _propose(self, name: str) -> None:
        self.deferred = {"id": "toolu_fake_001", "name": name, "input": {"value": "MCP_BRIDGE_OK"}}
        callback = self.initialize["hooks"]["PreToolUse"][0]["hookCallbackIds"][0]
        hook_input = {
            "hook_event_name": "PreToolUse",
            "tool_name": name,
            "tool_input": self.deferred["input"],
            "tool_use_id": self.deferred["id"],
            "session_id": "fixture",
            "transcript_path": "",
            "cwd": "/offline",
        }
        self.request(
            "proposal",
            {
                "subtype": "hook_callback",
                "callback_id": callback,
                "tool_use_id": self.deferred["id"],
                "input": hook_input,
            },
        )


class Harness:
    """Replaces the SDK client and query with protocol fakes, one per call."""

    def __init__(self) -> None:
        self.scenarios = ["text"]
        self.server_info: Frame = {
            "account": {"apiProvider": "firstParty", "subscriptionType": "Claude Team"},
            "claude_code_version": "2.1.312",
        }
        self.block_close = False
        self.transports: list[ProtocolTransport] = []
        self.preflights: list[ProtocolTransport] = []

    def client(self, options: ClaudeAgentOptions) -> ClaudeSDKClient:
        transport = ProtocolTransport(options, "preflight", self.server_info)
        self.preflights.append(transport)
        return ClaudeSDKClient(options=options, transport=transport)

    def query(
        self, prompt: AsyncIterable[Frame], options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        transport = ProtocolTransport(options, self.scenarios.pop(0), self.server_info)
        if self.block_close:
            transport.release_close.clear()
        self.transports.append(transport)
        return query(prompt=prompt, options=options, transport=transport)
