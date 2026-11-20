from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from mixhound.models.domain import DiscoveryReference, DiscoverySettings, Polarity

MAX_FEEDBACK_TEXT = 1000
MAX_FEEDBACK_TAGS = 20


class SessionIn(BaseModel):
    """A new session; without settings it starts from the default settings."""

    name: str = Field(min_length=1)
    instructions: str = ""
    references: list[DiscoveryReference] = Field(default_factory=list)
    settings: DiscoverySettings | None = None


class SessionNameIn(BaseModel):
    """A new session name."""

    name: str = Field(min_length=1)


class ReferenceIn(BaseModel):
    """A session reference."""

    kind: Literal["previous_result", "manual", "url"]
    artist: str = Field(min_length=1)
    title: str = Field(min_length=1)
    polarity: Polarity = "positive"
    priority: int = Field(default=3, ge=1, le=5)
    album: str | None = None
    year: int | None = None
    note: str = ""
    previous_track_id: UUID | None = None
    url: str | None = None
    version_qualifier: str | None = None


class ManualIn(BaseModel):
    """References typed as `Artist — Track` lines."""

    lines: str
    polarity: Polarity = "positive"


class RunRequest(BaseModel):
    """What a new run researches from."""

    instructions: str
    references: list[DiscoveryReference]
    settings: DiscoverySettings


class FeedbackIn(BaseModel):
    """A feedback action on a result."""

    action: Literal["mark", "skip", "comment", "tags", "exclude", "avoid_similar"]
    weight: int | None = Field(default=None, ge=1, le=5)
    polarity: Polarity | None = None
    text: str | None = Field(default=None, max_length=MAX_FEEDBACK_TEXT)
    tags: list[str] = Field(default_factory=list, max_length=MAX_FEEDBACK_TAGS)
    enabled: bool = True


class URLIn(BaseModel):
    """A public URL to read."""

    url: str
