import html
import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.evaluators.evaluator import EvaluatorOutput
from pydantic_evals.otel import SpanNode, SpanTree, SpanTreeRecordingError

from evals.cases import CaseMetadata, DiscoveryCase, RunOutcome
from mixhound.models.domain import DiscoveryRun, DiscoveryTrack
from mixhound.models.research import ResearchCandidate, ResearchOutput
from mixhound.services.identity import canonical_name, fold, normalize, recording_key

type Context = EvaluatorContext[DiscoveryCase, RunOutcome, CaseMetadata]

# Span attributes of pydantic-ai tool calls (instrumentation v2 names as fallback).
TOOL_NAME = "gen_ai.tool.name"
TOOL_ARGUMENTS = ("gen_ai.tool.call.arguments", "tool_arguments")
TOOL_RESULT = ("gen_ai.tool.call.result", "tool_response")
DEFERRED = "pydantic_ai.tool.deferral.name"
OUTPUT_FUNCTION = "running output function"
MAX_LISTED = 10


@dataclass(frozen=True)
class ToolCall:
    """One tool call an agent made, read from its span."""

    name: str
    arguments: str
    result: str


def _attribute(node: SpanNode, names: tuple[str, ...]) -> str:
    for name in names:
        value = node.attributes.get(name)
        if isinstance(value, str):
            return value
    return ""


def _is_tool_call(node: SpanNode) -> bool:
    name = node.attributes.get(TOOL_NAME)
    if not isinstance(name, str) or name.startswith("final_result") or DEFERRED in node.attributes:
        return False
    message = node.attributes.get("logfire.msg")
    if isinstance(message, str) and message.startswith(OUTPUT_FUNCTION):
        return False
    return node.name == "running tool" or node.name.startswith("execute_tool ")


def tool_calls(tree: SpanTree) -> list[ToolCall]:
    """Every tool call in the tree, in the order the calls started."""
    nodes = sorted((node for node in tree if _is_tool_call(node)), key=lambda n: n.start_timestamp)
    return [
        ToolCall(
            str(node.attributes[TOOL_NAME]),
            _attribute(node, TOOL_ARGUMENTS),
            _attribute(node, TOOL_RESULT),
        )
        for node in nodes
    ]


def _span_calls(ctx: Context) -> list[ToolCall] | None:
    """The case's tool calls, or None when its spans were not recorded."""
    try:
        return tool_calls(ctx.span_tree)
    except SpanTreeRecordingError:
        return None


def research_answers(diagnostic: Mapping[str, Any]) -> list[ResearchOutput]:
    """The research answers of a run that parse: the main one and each strategy's."""
    texts = [diagnostic.get("answer", "")]
    strategies = diagnostic.get("strategy_answers") or {}
    texts += list(strategies.values()) if isinstance(strategies, Mapping) else []
    answers: list[ResearchOutput] = []
    for text in texts:
        if not isinstance(text, str) or not text:
            continue
        try:
            answers.append(ResearchOutput.model_validate_json(text))
        except ValidationError:
            continue
    return answers


def _shown(run: DiscoveryRun) -> list[DiscoveryTrack]:
    return run.results + run.unverified_leads


def _label(track: DiscoveryTrack | ResearchCandidate) -> str:
    return f"{track.artist} — {track.title}"


def _listing(items: list[str]) -> str:
    more = f" (+{len(items) - MAX_LISTED} more)" if len(items) > MAX_LISTED else ""
    return "; ".join(items[:MAX_LISTED]) + more


def _check(offenders: list[str], rule: str) -> EvaluationReason:
    if not offenders:
        return EvaluationReason(value=True)
    return EvaluationReason(value=False, reason=f"{rule}: {_listing(offenders)}")


def _artist(name: str) -> str:
    return normalize(canonical_name(name))


@dataclass
class RunCompleted(Evaluator[DiscoveryCase, RunOutcome, CaseMetadata]):
    """The run finished with results rather than failing or being cancelled."""

    def evaluate(self, ctx: Context) -> EvaluationReason:
        """Pass when the run completed; the reason names a failure and a partial answer."""
        run = ctx.output.run
        detail = ctx.output.diagnostic.get("detail")
        notes = [f"failure {run.failure_code}" if run.failure_code else ""]
        notes += [f"detail {str(detail)[:200]}" if detail else "", "partial" if run.partial else ""]
        reason = ", ".join(note for note in notes if note) or None
        return EvaluationReason(value=run.status == "completed", reason=reason)


@dataclass
class VerifiedCount(Evaluator[DiscoveryCase, RunOutcome, CaseMetadata]):
    """How many verified results the run published, against the requested count."""

    def evaluate(self, ctx: Context) -> EvaluatorOutput:
        """Score the verified share and assert at least one verified result."""
        run = ctx.output.run
        desired = run.input.settings.desired_count
        return {
            "verified_share": round(len(run.results) / desired, 2),
            "any_verified": len(run.results) > 0,
        }


@dataclass
class ExclusionsRespected(Evaluator[DiscoveryCase, RunOutcome, CaseMetadata]):
    """Shown tracks obey the run's exclusions, year range and per-artist cap."""

    def evaluate(self, ctx: Context) -> EvaluatorOutput:
        """One assertion per rule; a failure lists the offending tracks."""
        run_input = ctx.output.run.input
        settings = run_input.settings
        shown = _shown(ctx.output.run)
        excluded = {normalize(name) for name in settings.excluded_artists}
        if settings.exclude_reference_artists:
            excluded |= {normalize(ref.artist) for ref in run_input.references}
        references = {
            recording_key(ref.artist, ref.title, qualifier=ref.version_qualifier)
            for ref in run_input.references
        }
        low, high = settings.year_from, settings.year_to
        per_artist = Counter(_artist(track.artist) for track in shown)
        return {
            "no_excluded_artist": _check(
                [_label(t) for t in shown if normalize(t.artist) in excluded], "Excluded artist"
            ),
            "no_reference_recording": _check(
                [
                    _label(t)
                    for t in shown
                    if recording_key(t.artist, t.title, qualifier=t.version_qualifier) in references
                ],
                "Reference recording",
            ),
            "within_years": _check(
                [
                    f"{_label(t)} ({t.year})"
                    for t in shown
                    if t.year is not None
                    and ((low is not None and t.year < low) or (high is not None and t.year > high))
                ],
                "Outside the year range",
            ),
            "per_artist_cap": _check(
                [
                    f"{artist}: {count}"
                    for artist, count in per_artist.items()
                    if count > settings.max_per_artist
                ],
                f"More than {settings.max_per_artist} per artist",
            ),
        }


@dataclass
class KnownReferenceIds(Evaluator[DiscoveryCase, RunOutcome, CaseMetadata]):
    """Answers and shown tracks cite only the run's reference IDs as closest references."""

    def evaluate(self, ctx: Context) -> EvaluationReason:
        """Fail with the unknown IDs and the tracks that cite them."""
        run = ctx.output.run
        known = {str(ref.id) for ref in run.input.references}
        cited = [
            (_label(candidate), ref_id)
            for answer in research_answers(ctx.output.diagnostic)
            for candidate in answer.candidates
            for ref_id in candidate.closest_reference_ids
        ]
        cited += [
            (_label(track), str(ref_id))
            for track in _shown(run)
            for ref_id in track.closest_reference_ids
        ]
        unknown = [f"{label}: {ref_id}" for label, ref_id in cited if ref_id not in known]
        return _check(list(dict.fromkeys(unknown)), "Unknown reference IDs")


@dataclass
class Novelty(Evaluator[DiscoveryCase, RunOutcome, CaseMetadata]):
    """How varied the verified results are and how far they move from the references."""

    def evaluate(self, ctx: Context) -> EvaluatorOutput:
        """Score distinct artists and artists that are not reference artists."""
        run = ctx.output.run
        artists = [_artist(track.artist) for track in run.results]
        if not artists:
            return {}
        references = {_artist(ref.artist) for ref in run.input.references}
        return {
            "distinct_artist_share": round(len(set(artists)) / len(artists), 2),
            "new_artist_share": round(
                sum(artist not in references for artist in artists) / len(artists), 2
            ),
        }


def comparable_url(url: str) -> str:
    """A URL without scheme, `www.`, fragment and trailing slash, lower-cased."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").removeprefix("www.")
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{parts.path.rstrip('/')}{query}".lower()


_SCHEME = re.compile(r"https?://(?:www\.)?", re.IGNORECASE)


def _answer_urls(answers: list[ResearchOutput]) -> Iterator[str]:
    for answer in answers:
        for candidate in answer.candidates:
            yield from (source.url for source in candidate.sources)
            yield from (path.source_url for path in candidate.discovery_paths if path.source_url)
            yield from candidate.listening_urls


def _fetched_url(call: ToolCall) -> str:
    try:
        arguments = json.loads(call.arguments) if call.arguments else {}
    except json.JSONDecodeError:
        return ""
    url = arguments.get("url") if isinstance(arguments, dict) else None
    return url if isinstance(url, str) else ""


@dataclass
class GroundedUrls(Evaluator[DiscoveryCase, RunOutcome, CaseMetadata]):
    """URLs in the research answers came from a tool rather than from the model's memory.

    A URL counts as grounded when a tool result contains it or the agent opened it.
    """

    def evaluate(self, ctx: Context) -> EvaluatorOutput:
        """Score the grounded share; the reason lists URLs no tool returned."""
        calls = _span_calls(ctx)
        urls = list(dict.fromkeys(_answer_urls(research_answers(ctx.output.diagnostic))))
        if calls is None or not urls:
            return {}
        seen = "\n".join(
            [html.unescape(call.result) for call in calls]
            + [_fetched_url(call) for call in calls if call.name == "fetch_page"]
        )
        seen = _SCHEME.sub("", seen).lower()
        guessed = [url for url in urls if comparable_url(url) not in seen]
        share = round(1 - len(guessed) / len(urls), 2)
        reason = f"Not returned by any tool: {_listing(guessed)}" if guessed else None
        return {"grounded_url_share": EvaluationReason(value=share, reason=reason)}


def _candidate_key(artist: object, title: object) -> tuple[str, str]:
    return fold(str(artist)), fold(str(title))


def _checked(calls: list[ToolCall]) -> set[tuple[str, str]]:
    """Recordings passed to `verify_candidates`."""
    keys: set[tuple[str, str]] = set()
    for call in calls:
        if call.name != "verify_candidates" or not call.arguments:
            continue
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError:
            continue
        candidates = arguments.get("candidates") if isinstance(arguments, dict) else None
        keys.update(
            _candidate_key(item.get("artist"), item.get("title"))
            for item in candidates or []
            if isinstance(item, dict)
        )
    return keys


@dataclass
class CatalogCheckedBeforeAnswer(Evaluator[DiscoveryCase, RunOutcome, CaseMetadata]):
    """In agentic mode, the orchestrator checked its candidates against catalogs before answering.

    Pipeline mode verifies every candidate on the server, so it is not scored.
    """

    def evaluate(self, ctx: Context) -> EvaluatorOutput:
        """Assert a catalog check happened and score the share of answered candidates checked."""
        calls = _span_calls(ctx)
        if calls is None or ctx.output.run.input.settings.mode != "agentic":
            return {}
        checked = _checked(calls)
        result: dict[str, EvaluationReason | float | bool] = {
            "catalog_checked": any(call.name == "verify_candidates" for call in calls)
        }
        answers = research_answers({"answer": ctx.output.diagnostic.get("answer", "")})
        candidates = [c for answer in answers for c in answer.candidates]
        if candidates:
            unchecked = [
                _label(c) for c in candidates if _candidate_key(c.artist, c.title) not in checked
            ]
            reason = f"Answered unchecked: {_listing(unchecked)}" if unchecked else None
            share = round(1 - len(unchecked) / len(candidates), 2)
            result["checked_share"] = EvaluationReason(value=share, reason=reason)
        return result


@dataclass
class ToolUsage(Evaluator[DiscoveryCase, RunOutcome, CaseMetadata]):
    """How often the agents called each tool."""

    def evaluate(self, ctx: Context) -> EvaluatorOutput:
        """Count tool calls by name, plus the total."""
        calls = _span_calls(ctx)
        if calls is None:
            return {}
        counts = Counter(call.name for call in calls)
        return {"tool_calls": len(calls)} | {f"calls_{name}": n for name, n in counts.items()}


def default_evaluators() -> list[Evaluator[DiscoveryCase, RunOutcome, CaseMetadata]]:
    """The rule-based evaluators every eval run applies."""
    return [
        RunCompleted(),
        VerifiedCount(),
        ExclusionsRespected(),
        KnownReferenceIds(),
        Novelty(),
        GroundedUrls(),
        CatalogCheckedBeforeAnswer(),
        ToolUsage(),
    ]
