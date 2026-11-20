from dataclasses import dataclass, field
from uuid import UUID

from mixhound.services.audio_features import AudioFeatures


@dataclass
class ReferenceFacts:
    """What the server learned about one reference before research."""

    mbid: str | None = None
    artist_mbids: tuple[str, ...] = ()
    tags: list[str] = field(default_factory=list)
    audio: AudioFeatures | None = None
    # The catalog page whose preview clip measured the reference.
    preview_url: str | None = None


@dataclass
class Preparation:
    """Facts about every reference and the candidate pool for the research."""

    facts: dict[UUID, ReferenceFacts] = field(default_factory=dict)
    pool: list[dict[str, object]] = field(default_factory=list)
