from typing import Literal

from pydantic import BaseModel, Field

from mixhound.models.domain import SonicProfile


class ResearchSource(BaseModel):
    """A source as the research agent reports it."""

    url: str
    claim: str
    target_type: Literal["track", "album", "artist", "review", "other"]


class ResearchPath(BaseModel):
    """A discovery path as the research agent reports it."""

    strategy: str
    explanation: str
    source_url: str | None


class ResearchCandidate(BaseModel):
    """One candidate as the research agent reports it: identity, claims, and URLs only.

    Reachability, identity confirmation, playback, and the verified/unverified decision are
    established by the server, so the model is never asked for them.
    """

    artist: str
    title: str
    artist_alt: str | None
    title_alt: str | None
    album: str | None
    year: int | None
    version_qualifier: str | None
    original_artist: str | None
    original_title: str | None
    genres: list[str]
    similarity_score: int | None = Field(ge=0, le=100)
    confidence: Literal["low", "medium", "high"]
    # False for a real recording seen in passing (a search result, a track list) and returned
    # without its own research.
    researched: bool
    similarities: list[str]
    differences: list[str]
    closest_reference_ids: list[str]
    discovery_paths: list[ResearchPath]
    sources: list[ResearchSource]
    listening_urls: list[str]


class ResearchOutput(BaseModel):
    """The research agent's structured answer."""

    sonic_profile: SonicProfile
    candidates: list[ResearchCandidate]


class FoundLink(BaseModel):
    """A track page found by the link follow-up."""

    artist: str
    title: str
    url: str


class LinkSearchOutput(BaseModel):
    """The listening-link follow-up's structured answer."""

    links: list[FoundLink]
