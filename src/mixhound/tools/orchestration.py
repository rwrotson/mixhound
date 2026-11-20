import json
from typing import Literal

from pydantic_ai import RunContext
from pydantic_ai.framing import register_tag, wrap

from mixhound.models.domain import SonicProfile
from mixhound.models.research import ResearchCandidate
from mixhound.services.agentic import (
    CandidateReport,
    DelegateReport,
    ReferenceReport,
    delegate_limits,
)
from mixhound.tools.deps import ToolDeps

# Catalog facts and leads come from third parties: data, never instructions.
CATALOG_FACTS = register_tag("catalog_facts")
# Delegated candidates listed back to the orchestrator.
SHOWN_CANDIDATES = 10
NOT_AGENTIC = "This tool is only available in agentic runs."

type Strategy = Literal["collaborators", "label_scene", "traits"]


def _reference_text(report: ReferenceReport) -> str:
    reference = report.reference
    lines = [f"Reference {reference.id}: {reference.artist} — {reference.title}"]
    if report.catalog_down:
        lines.append("Catalog: unavailable right now.")
    elif not report.found:
        lines.append("Catalog: not found.")
    else:
        mbid = f" (MusicBrainz {report.facts.mbid})" if report.facts.mbid else ""
        lines.append(f"Catalog: found{mbid}.")
    if report.facts.tags:
        lines.append(f"Tags: {', '.join(report.facts.tags)}")
    if report.facts.audio is not None:
        lines.append(f"Measured audio: {json.dumps(report.facts.audio.describe())}")
    lines.append(f"New candidate pool leads ({len(report.leads)}):")
    lines += [
        f"- {lead['artist']} — {lead['title'] or '(artist)'}; {lead['via']}"
        + (f"; {lead['url']}" if lead.get("url") else "")
        for lead in report.leads
    ]
    return wrap(CATALOG_FACTS, "\n".join(lines))


async def lookup_reference(ctx: RunContext[ToolDeps], reference_id: str) -> str:
    """Look a reference up in the music catalogs: tags, measured audio and candidate leads.

    A positive reference adds its share of leads (similar recordings and artists,
    collaborators) to the candidate pool. Repeating a lookup returns the same answer.

    Args:
        reference_id: The reference's `id` from the run data.
    """
    deps = ctx.deps
    workspace = deps.workspace
    if workspace is None:
        return NOT_AGENTIC
    reference = workspace.reference(reference_id.strip())
    if reference is None:
        return f"No positive or negative reference has the ID {reference_id}."
    if not deps.take("lookup_reference"):
        return "lookup_reference limit reached for this run."
    return _reference_text(await workspace.lookup_reference(reference))


def _candidate_text(number: int, report: CandidateReport) -> str:
    head = f"{number}. {report.artist} — {report.title}: "
    if report.confirmed:
        year = f", year {report.year}" if report.year else ""
        parts = [f"confirmed by {', '.join(report.confirmed_by)}{year}"]
    else:
        parts = ["not confirmed by the catalogs"]
    checked = [
        f"{source.url} {'names it' if source.matches else 'does not name it'}"
        for source in report.sources
    ]
    if checked:
        parts.append("source pages: " + "; ".join(checked))
    if report.listening_urls:
        parts.append("track pages: " + ", ".join(report.listening_urls))
    if report.audio_similarity is not None:
        parts.append(f"audio similarity {report.audio_similarity:.2f} ({report.audio_evidence})")
    return head + "; ".join(parts) + "."


async def verify_candidates(ctx: RunContext[ToolDeps], candidates: list[ResearchCandidate]) -> str:
    """Check candidates as the server will before publishing: catalogs, source pages, audio.

    Only confirmed recordings can be published as verified results.

    Args:
        candidates: Candidates with the same fields as in the final answer.
    """
    deps = ctx.deps
    workspace = deps.workspace
    if workspace is None:
        return NOT_AGENTIC
    room = deps.room("verify_candidates")
    if not candidates:
        return "No candidates given."
    if room == 0:
        return "verify_candidates limit reached for this run; answer with what you have."
    batch = candidates[:room]
    deps.take("verify_candidates", len(batch))
    reports = await workspace.verify(batch)
    text = "\n".join(_candidate_text(number, report) for number, report in enumerate(reports, 1))
    skipped = len(candidates) - len(batch)
    if skipped:
        text += f"\n{skipped} candidate(s) not checked: verify_candidates limit reached."
    return text


def _delegate_text(report: DelegateReport) -> str:
    if not report.candidates:
        reason = report.failure or "no candidates"
        return f"Strategy {report.strategy} returned nothing: {reason}."
    partial = f" (cut off: {report.failure})" if report.partial else ""
    best = sorted(report.candidates, key=lambda item: -(item.similarity_score or 0))
    shown = "\n".join(
        f"- {item.artist} — {item.title} ({item.similarity_score}, {item.confidence})"
        for item in best[:SHOWN_CANDIDATES]
    )
    return (
        f"Strategy {report.strategy} kept {len(report.candidates)} candidates{partial}; the "
        f"server merges them with your answer, so repeat one only to improve it. Best:\n{shown}"
    )


async def delegate_research(
    ctx: RunContext[ToolDeps], strategy: Strategy, profile: SonicProfile, brief: str = ""
) -> str:
    """Hand one research strategy to a separate researcher; several calls run in parallel.

    collaborators: credited collaborators, their projects and aliases. label_scene: labels,
    compilations, radio shows and mixes. traits: searches for the profile's traits, covers,
    remixes and samples.

    Args:
        strategy: The strategy to delegate; each one at most once.
        profile: The sonic profile the researcher should match.
        brief: Optional directions: paths to follow or avoid, what you already cover.
    """
    deps = ctx.deps
    workspace = deps.workspace
    if workspace is None:
        return NOT_AGENTIC
    if not deps.take("delegate_research"):
        return "delegate_research limit reached for this run."
    cost = ctx.usage.cost
    limits = delegate_limits(
        deps.state, deps.elapsed(), deps.call.seconds, float(cost) if cost is not None else None
    )
    return _delegate_text(await workspace.delegate(strategy, profile, brief, limits))
