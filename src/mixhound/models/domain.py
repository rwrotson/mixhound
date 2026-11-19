from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

# Whether a reference or a marked result is an example to follow, one to avoid, or off.
type Polarity = Literal["positive", "negative", "neutral"]


def new_id() -> UUID:
    """Generate a random record ID."""
    return uuid4()


def now_utc() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


class Record(BaseModel):
    """Base of persisted Discovery records, carrying the schema version."""

    version: Literal[1] = 1


class DatedRecord(Record):
    """A record with an ID and a creation time."""

    id: UUID = Field(default_factory=new_id)
    created_at: datetime = Field(default_factory=now_utc)

    @field_validator("created_at")
    @classmethod
    def utc_time(cls, value: datetime) -> datetime:
        """Require a timezone-aware timestamp and convert it to UTC."""
        if value.tzinfo is None:
            raise ValueError("Timestamp must have a timezone")
        return value.astimezone(UTC)


NEUTRAL_RATING = 3
MODEL_PATTERN = r"^[A-Za-z0-9._:/@\[\]-]{1,120}$"


class DiscoverySettings(Record):
    """Research settings of a session or a run: mode, model, depth, limits and filters."""

    # "pipeline" runs the V1 server-driven stages; "agentic" lets an orchestrator agent choose them.
    mode: Literal["pipeline", "agentic"] = "pipeline"
    # A pydantic-ai model string such as "claude-code:sonnet"; None uses the configured default.
    model: str | None = Field(default=None, pattern=MODEL_PATTERN)
    desired_count: int = Field(default=10, ge=1, le=30)
    # Candidate ceiling and size of the server's lead pool; time and money are separate.
    depth: Literal["quick", "standard", "deep"] = "standard"
    # Wall-clock limit of the research agent, which the model also sees as a clock.
    minutes: int = Field(default=10, ge=1, le=60)
    # Estimated spending cap of the research agent, where model prices are known.
    max_usd: float = Field(default=3.0, gt=0, le=50)
    # None leaves the effort to the model's own default.
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max", "ultra"] | None = "high"
    # Cheaper model and effort for auxiliary stages (the listening-link follow-up).
    helper_model: str | None = Field(default=None, pattern=MODEL_PATTERN)
    helper_reasoning_effort: Literal["low", "medium", "high", "xhigh", "max", "ultra"] | None = (
        "low"
    )
    max_per_artist: int = Field(default=2, ge=1, le=10)
    year_from: int | None = None
    year_to: int | None = None
    excluded_artists: list[str] = Field(default_factory=list)
    exclude_reference_artists: bool = False
    exclude_previously_proposed_artists: bool = False
    # Experimental: research three discovery strategies in parallel after one profile pass.
    parallel_strategies: bool = False

    @field_validator("model", "helper_model", mode="before")
    @classmethod
    def blank_model(cls, value: object) -> object:
        """Treat a blank model name as the configured default."""
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @model_validator(mode="after")
    def valid_years(self) -> DiscoverySettings:
        """Reject a year range that starts after it ends."""
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("Starting year must not exceed ending year")
        return self


class DiscoveryReference(DatedRecord):
    """A recording a session researches from, with its polarity and priority."""

    kind: Literal["previous_result", "manual", "url"]
    artist: str
    title: str
    polarity: Polarity = "positive"
    priority: int = Field(default=3, ge=1, le=5)
    album: str | None = None
    year: int | None = None
    note: str = ""
    previous_track_id: UUID | None = None
    url: str | None = None
    version_qualifier: str | None = None


class DiscoverySource(Record):
    """A page that supports a claim about a result."""

    url: str
    source_type: str
    claim: str
    target_type: Literal["track", "album", "artist", "review", "other"]
    reachable: bool | None = None
    matches_recording: bool | None = None


class DiscoveryListeningLink(Record):
    """A page where a result can be listened to, with its checked availability."""

    url: str
    service: str
    target_type: Literal["track", "album", "other"]
    reachable: bool | None = None
    playback_available: bool | None = None


class DiscoveryPath(Record):
    """How the research reached a result."""

    strategy: str
    explanation: str
    source_url: str | None = None


class DiscoveryTrack(DatedRecord):
    """A recording recommended by a run, with its evidence, links and scores."""

    artist: str
    title: str
    # Transliteration or translation of a name written in another script.
    artist_alt: str | None = None
    title_alt: str | None = None
    isrc: str | None = None
    album: str | None = None
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    version_qualifier: str | None = None
    original_artist: str | None = None
    original_title: str | None = None
    external_recording_id: str | None = None
    similarity_score: int | None = Field(default=None, ge=0, le=100)
    confidence: Literal["low", "medium", "high"] = "low"
    audio_analyzed: bool = False
    audio_analysis_evidence: str | None = None
    # Feature-based similarity of a short audio clip to the references, 0-100.
    audio_similarity: int | None = Field(default=None, ge=0, le=100)
    # The clip's measured features, kept so a marked result can be an audio reference later.
    audio_features: dict[str, float] | None = None
    verification: Literal["verified", "unverified"] = "unverified"
    # False for a recording the research saw but did not study: never published as verified.
    researched: bool = True
    similarities: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    closest_reference_ids: list[UUID] = Field(default_factory=list)
    sources: list[DiscoverySource] = Field(default_factory=list)
    listening_links: list[DiscoveryListeningLink] = Field(default_factory=list)
    preferred_listening_url: str | None = None
    discovery_paths: list[DiscoveryPath] = Field(default_factory=list)


class SonicProfile(BaseModel):
    """What the references share, derived before searching and reused by later runs."""

    summary: str = ""
    shared_traits: list[str] = Field(default_factory=list)
    avoid_traits: list[str] = Field(default_factory=list)


class DiscoveryResearchResult(Record):
    """Server-side container of one run's candidates while they are verified."""

    results: list[DiscoveryTrack] = Field(default_factory=list)
    unverified_leads: list[DiscoveryTrack] = Field(default_factory=list)
    sonic_profile: SonicProfile | None = None


class StrategyMetrics(BaseModel):
    """One parallel strategy's wall time, yield, and spend."""

    seconds: float = 0.0
    candidates: int = 0
    cost_usd: float | None = None
    web_searches: int = 0
    failed: str | None = None  # Failure code; None when the strategy answered.


class RunMetrics(Record):
    """Where a run spent its time and money."""

    stages: dict[str, float] = Field(default_factory=dict)
    # API-equivalent price estimate where model prices are known. With a subscription login
    # nothing is charged: it measures the share of the usage limit taken.
    cost_usd: float | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    web_searches: int = 0
    # Searches made by the research itself (single pass or strategies), not by helpers.
    research_searches: int = 0
    turns: int = 0
    llm_calls: int = 0
    candidates: int = 0
    pool_size: int = 0
    verified: int = 0
    leads: int = 0
    # Leads the research returned without studying them.
    unresearched: int = 0
    dropped: int = 0
    playable: int = 0
    audio_analyzed: int = 0
    # Candidates citing a source that describes their sound (a review, release or label text).
    sonic_evidence: int = 0
    cache_hits: int = 0
    # Page fetches by the research that failed (403, 404, blocked or too large).
    failed_fetches: int = 0
    # Parallel strategies only: per-strategy figures and candidates found by two or more.
    strategies: dict[str, StrategyMetrics] = Field(default_factory=dict)
    overlap: int = 0


class DiscoveryRunInput(Record):
    """What a run starts from: references, instructions, feedback and settings."""

    feedback: list[dict[str, object]] = Field(default_factory=list)
    references: list[DiscoveryReference] = Field(default_factory=list)
    instructions: str = ""
    message: str = ""
    settings: DiscoverySettings = Field(default_factory=DiscoverySettings)
    refinement: bool = False


class DiscoveryRun(DatedRecord):
    """One research run: its input, status, results and metrics."""

    session_id: UUID
    input: DiscoveryRunInput
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    finished_at: datetime | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    results: list[DiscoveryTrack] = Field(default_factory=list)
    unverified_leads: list[DiscoveryTrack] = Field(default_factory=list)
    sonic_profile: SonicProfile | None = None
    metrics: RunMetrics | None = None
    # Research stopped at its time or cost limit; results come from its interim notes.
    partial: bool = False

    @field_validator("finished_at")
    @classmethod
    def utc_finish(cls, value: datetime | None) -> datetime | None:
        """Normalize the finish time to UTC."""
        return DatedRecord.utc_time(value) if value is not None else None


class FeedbackEvent(DatedRecord):
    """One feedback action on a run result."""

    run_id: UUID
    track_id: UUID
    # "rating" is the legacy 1-5 quality score; it is only replayed, through `legacy_mark`.
    action: Literal["mark", "rating", "skip", "comment", "tags", "exclude", "avoid_similar"]
    rating: int | None = Field(default=None, ge=1, le=5)
    # A mark: weight 1 (minor) to 5 (decisive), like reference priority, and its sign.
    weight: int | None = Field(default=None, ge=1, le=5)
    polarity: Polarity | None = None
    text: str | None = None
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True

    @model_validator(mode="after")
    def valid_action(self) -> FeedbackEvent:
        """Require the fields each action needs."""
        if self.action == "rating" and self.rating is None:
            raise ValueError("Rating action requires a rating")
        if self.action == "mark" and (
            self.polarity is None or (self.polarity != "neutral" and self.weight is None)
        ):
            raise ValueError("Mark action requires a polarity, and a weight unless neutral")
        return self


def legacy_mark(rating: int) -> tuple[Polarity, int]:
    """A legacy quality rating as a mark: 1-2 negative, 3 off, 4-5 positive; extremes weigh 5."""
    polarity: Polarity = (
        "negative"
        if rating < NEUTRAL_RATING
        else "positive"
        if rating > NEUTRAL_RATING
        else "neutral"
    )
    return polarity, 5 if rating in (1, 5) else 3


# Feedback actions that switch a flag of the feedback state on or off.
_FLAG_ACTIONS = {"skip": "skipped", "exclude": "excluded", "avoid_similar": "avoid_similar"}


class FeedbackState(Record):
    """A result's current feedback, replayed from its events."""

    weight: int | None = None
    polarity: Polarity = "neutral"
    skipped: bool = False
    text: str | None = None
    tags: list[str] = Field(default_factory=list)
    excluded: bool = False
    avoid_similar: bool = False

    def apply(self, event: FeedbackEvent) -> None:
        """Update the state with one feedback event."""
        if event.action in _FLAG_ACTIONS:
            setattr(self, _FLAG_ACTIONS[event.action], event.enabled)
        elif event.action == "comment":
            self.text = event.text
        elif event.action == "tags":
            self.tags = event.tags.copy()
        # The model validator requires a polarity for marks and a rating for ratings.
        elif event.action == "mark" and event.polarity is not None:
            self.polarity, self.weight, self.skipped = event.polarity, event.weight, False
        elif event.action == "rating" and event.rating is not None:
            (self.polarity, self.weight), self.skipped = legacy_mark(event.rating), False


class DiscoverySession(DatedRecord):
    """A named set of references, instructions and settings, with its runs and feedback."""

    name: str
    references: list[DiscoveryReference] = Field(default_factory=list)
    instructions: str = ""
    settings: DiscoverySettings = Field(default_factory=DiscoverySettings)
    run_ids: list[UUID] = Field(default_factory=list)
    feedback_events: list[FeedbackEvent] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=now_utc)

    @field_validator("updated_at")
    @classmethod
    def utc_update(cls, value: datetime) -> datetime:
        """Normalize the update time to UTC."""
        return DatedRecord.utc_time(value)

    def feedback_for(self, track_id: UUID) -> FeedbackState:
        """Replay the feedback events of one result into its current state."""
        state = FeedbackState()
        for event in self.feedback_events:
            if event.track_id == track_id:
                state.apply(event)
        return state


class DiscoveryIndex(Record):
    """The Discovery index: default settings and session order."""

    settings: DiscoverySettings = Field(default_factory=DiscoverySettings)
    session_ids: list[UUID] = Field(default_factory=list)
