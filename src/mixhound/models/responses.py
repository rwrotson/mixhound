from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from mixhound.models.domain import Polarity, SonicProfile, StrategyMetrics

type Effort = Literal["low", "medium", "high", "xhigh", "max", "ultra"]


class ApiOut(BaseModel):
    """Base of responses mapped from stored records, without their storage internals."""

    model_config = ConfigDict(extra="ignore", from_attributes=True)


class SettingsOut(ApiOut):
    """Research settings."""

    mode: Literal["pipeline", "agentic"]
    model: str | None
    desired_count: int
    depth: Literal["quick", "standard", "deep"]
    minutes: int
    max_usd: float
    reasoning_effort: Effort | None
    helper_model: str | None
    helper_reasoning_effort: Effort | None
    max_per_artist: int
    year_from: int | None
    year_to: int | None
    excluded_artists: list[str]
    exclude_reference_artists: bool
    exclude_previously_proposed_artists: bool
    parallel_strategies: bool


class ReferenceOut(ApiOut):
    """A session reference."""

    id: UUID
    created_at: datetime
    kind: Literal["previous_result", "manual", "url"]
    artist: str
    title: str
    polarity: Polarity
    priority: int
    album: str | None
    year: int | None
    note: str
    previous_track_id: UUID | None
    url: str | None
    version_qualifier: str | None


class FeedbackEventOut(ApiOut):
    """One feedback action on a run result."""

    id: UUID
    created_at: datetime
    run_id: UUID
    track_id: UUID
    action: Literal["mark", "rating", "skip", "comment", "tags", "exclude", "avoid_similar"]
    rating: int | None
    weight: int | None
    polarity: Polarity | None
    text: str | None
    tags: list[str]
    enabled: bool


class SessionOut(ApiOut):
    """A session with its references, settings, runs and feedback."""

    id: UUID
    created_at: datetime
    name: str
    references: list[ReferenceOut]
    instructions: str
    settings: SettingsOut
    run_ids: list[UUID]
    feedback_events: list[FeedbackEventOut]
    updated_at: datetime


class SourceOut(ApiOut):
    """A page supporting a claim about a result."""

    url: str
    source_type: str
    claim: str
    target_type: Literal["track", "album", "artist", "review", "other"]
    reachable: bool | None
    matches_recording: bool | None


class ListeningLinkOut(ApiOut):
    """A page where a result can be listened to; playback is checked by the client."""

    url: str
    service: str
    target_type: Literal["track", "album", "other"]
    reachable: bool | None
    playback_available: bool | None


class PathOut(ApiOut):
    """How the research reached a result."""

    strategy: str
    explanation: str
    source_url: str | None


class ResultTrackOut(ApiOut):
    """A recommended recording, without its measured clip features."""

    id: UUID
    created_at: datetime
    artist: str
    title: str
    artist_alt: str | None
    title_alt: str | None
    isrc: str | None
    album: str | None
    year: int | None
    genres: list[str]
    version_qualifier: str | None
    original_artist: str | None
    original_title: str | None
    external_recording_id: str | None
    similarity_score: int | None
    confidence: Literal["low", "medium", "high"]
    audio_analyzed: bool
    audio_analysis_evidence: str | None
    audio_similarity: int | None
    verification: Literal["verified", "unverified"]
    researched: bool
    similarities: list[str]
    differences: list[str]
    closest_reference_ids: list[UUID]
    sources: list[SourceOut]
    listening_links: list[ListeningLinkOut]
    preferred_listening_url: str | None
    discovery_paths: list[PathOut]


class RunInputOut(ApiOut):
    """What a run started from."""

    feedback: list[dict[str, object]]
    references: list[ReferenceOut]
    instructions: str
    message: str
    settings: SettingsOut
    refinement: bool


class RunMetricsOut(ApiOut):
    """Where a run spent its time and money."""

    stages: dict[str, float]
    cost_usd: float | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    web_searches: int
    research_searches: int
    turns: int
    llm_calls: int
    candidates: int
    pool_size: int
    verified: int
    leads: int
    unresearched: int
    dropped: int
    playable: int
    audio_analyzed: int
    sonic_evidence: int
    cache_hits: int
    failed_fetches: int
    strategies: dict[str, StrategyMetrics]
    overlap: int


class RunOut(ApiOut):
    """One research run."""

    id: UUID
    created_at: datetime
    session_id: UUID
    input: RunInputOut
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    finished_at: datetime | None
    failure_code: str | None
    failure_message: str | None
    results: list[ResultTrackOut]
    unverified_leads: list[ResultTrackOut]
    sonic_profile: SonicProfile | None
    metrics: RunMetricsOut | None
    partial: bool


class FeedbackStateOut(ApiOut):
    """A result's current feedback."""

    weight: int | None
    polarity: Polarity
    skipped: bool
    text: str | None
    tags: list[str]
    excluded: bool
    avoid_similar: bool


class ResolvedUrlOut(ApiOut):
    """The recording a public page describes, as far as its metadata says."""

    url: str
    service: str
    artist: str | None
    title: str | None


class ProvidersOut(BaseModel):
    """What this process is configured to use; no credential is checked or revealed."""

    model: str
    helper_model: str | None
    fallback_model: str | None
    web_search: bool
    audio: bool
    # Whether each optional deadwax source has its key.
    catalog_keys: dict[str, bool]
