# Vendored from h0rv/pydantic-ai-claude-code (MIT, (c) 2026 Armin Ronacher, chem), adapted.
import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import replace

from pydantic import JsonValue, TypeAdapter
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.tools import ToolDefinition

_OBJECT = TypeAdapter(dict[str, JsonValue])
_OBJECTS = TypeAdapter(list[dict[str, JsonValue]])
_ALIAS = re.compile(r"[a-zA-Z0-9_-]{1,48}")
MCP_PREFIX = "mcp__pydantic_ai__"
CLI_HEADERS = frozenset(
    {
        "authorization",
        "x-api-key",
        "anthropic-version",
        "host",
        "user-agent",
        "content-type",
        "content-length",
        "accept",
        "accept-encoding",
        "connection",
        "x-app",
        "x-claude-code-session-id",
    }
)


def application_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return headers the application may add; Claude Code owns auth and protocol headers."""
    result: list[tuple[str, str]] = []
    for name, value in headers:
        if name.lower() in CLI_HEADERS or name.lower().startswith("x-stainless-"):
            raise UserError("Claude Code owns this authentication/protocol header: " + name)
        if any(character in name + value for character in ("\r", "\n")):
            raise UserError("HTTP headers cannot contain carriage returns or newlines.")
        result.append((name, value))
    return result


def object_value(value: JsonValue) -> dict[str, JsonValue]:
    """Return a protocol value that must be a JSON object."""
    return _OBJECT.validate_python(value, strict=True)


def text_value(value: JsonValue) -> str:
    """Return a protocol value that must be a string."""
    if not isinstance(value, str):
        raise UnexpectedModelBehavior("Expected a protocol string.")
    return value


def catalog(tools: list[ToolDefinition]) -> tuple[list[ToolDefinition], dict[str, str]]:
    """Give tools MCP-safe aliases; return them and a map from MCP name to original name."""
    aliased: list[ToolDefinition] = []
    names: dict[str, str] = {}
    for tool in sorted(tools, key=lambda item: item.name):
        alias = (
            tool.name
            if _ALIAS.fullmatch(tool.name)
            else "tool_" + hashlib.sha256(tool.name.encode()).hexdigest()[:32]
        )
        transport_name = MCP_PREFIX + alias
        if transport_name in names:
            raise UserError("Tool names collide after MCP normalization.")
        names[transport_name] = tool.name
        aliased.append(replace(tool, name=alias))
    return aliased, names


def tool_blocks(content: JsonValue) -> Iterator[dict[str, JsonValue]]:
    """Yield every block of message content that names a tool, nested ones included."""
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind in ("tool_use", "tool_reference"):
            yield block
        elif kind in ("tool_addition", "tool_removal"):
            yield from tool_blocks([block.get("tool")])
        elif kind == "tool_result":
            yield from tool_blocks(block.get("content"))


def tool_name_key(block: dict[str, JsonValue]) -> str:
    """Return the key that holds the tool name in a block."""
    return "tool_name" if block.get("type") == "tool_reference" and "tool_name" in block else "name"


def _definitions(tools: list[dict[str, JsonValue]]) -> list[ToolDefinition]:
    definitions: list[ToolDefinition] = []
    for tool in tools:
        if tool.get("type", "custom") != "custom":
            raise UserError("Claude Code relay does not support provider-native tools.")
        definitions.append(
            ToolDefinition(
                name=text_value(tool.get("name")),
                description=text_value(tool.get("description", "")),
                parameters_json_schema=object_value(tool.get("input_schema")),
            )
        )
    return definitions


def prepare(
    body: dict[str, JsonValue],
) -> tuple[dict[str, JsonValue], list[ToolDefinition], dict[str, str]]:
    """Rename tools to MCP aliases in a native request; return it, the catalog and name map."""
    tools = _OBJECTS.validate_python(body.get("tools", []), strict=True)
    definitions = _definitions(tools)
    manifest, names = catalog(definitions)
    messages = _OBJECTS.validate_python(body.get("messages"), strict=True)
    historical = {
        text_value(block.get(tool_name_key(block)))
        for message in messages
        for block in tool_blocks(message.get("content"))
    }
    every = historical | {definition.name for definition in definitions}
    _, all_names = catalog([ToolDefinition(name=name) for name in every])
    forward = {original: alias for alias, original in all_names.items()}
    for tool in tools:
        tool["name"] = forward[text_value(tool.get("name"))]
    for message in messages:
        for block in tool_blocks(message.get("content")):
            key = tool_name_key(block)
            block[key] = forward[text_value(block.get(key))]
    result: dict[str, JsonValue] = {**body, "messages": [*messages], "stream": True}
    if "tools" in body:
        result["tools"] = [*tools]
    choice = result.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "tool":
        choice["name"] = forward[text_value(choice.get("name"))]
    return result, manifest, names


def system_blocks(value: JsonValue) -> list[dict[str, JsonValue]]:
    """Return the native system prompt as text blocks."""
    if value is None:
        return []
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    return _OBJECTS.validate_python(value, strict=True)


def extra_body(desired: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Supply application API options before the CLI constructs its native body."""
    if desired.get("metadata"):
        raise UserError(
            "Claude Code owns request metadata for subscription attribution; "
            "it cannot be overridden."
        )
    body = {
        key: value
        for key, value in desired.items()
        if key not in {"model", "messages", "system", "metadata"}
    }
    body.setdefault("tools", [])
    body.setdefault("thinking", {"type": "disabled"})
    if object_value(body["thinking"]).get("type") == "disabled":
        body.setdefault("context_management", {"edits": []})
    return body


def _blocks(message: dict[str, JsonValue]) -> tuple[str, list[JsonValue]]:
    role = text_value(message.get("role"))
    if role not in ("user", "assistant"):
        raise UserError(
            "Claude Code history must contain native user, assistant, or system messages."
        )
    content = message.get("content")
    blocks: JsonValue = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(blocks, list):
        raise UserError("Claude Code message content must be text or native content blocks.")
    return role, blocks


def _append(frames: list[dict[str, JsonValue]], message: dict[str, JsonValue]) -> None:
    """Add a message as a frame; consecutive user messages share one frame."""
    role, blocks = _blocks(message)
    if not blocks:
        return
    if role == "user" and frames and frames[-1]["type"] == "user":
        previous = object_value(frames[-1]["message"])
        previous_blocks = previous["content"]
        if not isinstance(previous_blocks, list):
            raise UnexpectedModelBehavior("Expected native content blocks.")
        previous_blocks.extend(blocks)
        frames[-1]["message"] = previous
    else:
        frames.append({"type": role, "message": {**message, "content": blocks}})


def input_frames(desired: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    """Replay the history as SDK input frames; only the last user frame asks for inference."""
    frames: list[dict[str, JsonValue]] = []
    for message in _OBJECTS.validate_python(desired.get("messages"), strict=True):
        if message.get("role") != "system":
            _append(frames, message)
    if not frames or frames[-1]["type"] != "user":
        raise UserError("Claude Code history must end in a non-empty user or tool-result message.")
    for frame in frames[:-1]:
        if frame["type"] == "user":
            frame["shouldQuery"] = False
    return frames


def system_prompt(desired: dict[str, JsonValue]) -> str:
    """Return the native system prompt as plain text."""
    blocks = system_blocks(desired.get("system"))
    if any(set(block) - {"type", "text"} for block in blocks):
        raise UserError(
            "Claude Code owns native system-block caching; "
            "per-block system options are unsupported."
        )
    return "\n".join(text_value(block.get("text")) for block in blocks)
