from dataclasses import dataclass

from mixhound.models.domain import DiscoveryTrack
from mixhound.services.identity import recording_key


def metadata_key(track: DiscoveryTrack) -> str:
    """Return the metadata identity key of a result."""
    return recording_key(track.artist, track.title, qualifier=track.version_qualifier)


@dataclass(frozen=True)
class Exclusions:
    """History-derived keys a new recommendation must not repeat."""

    prior_keys: frozenset[str] = frozenset()
    prior_artists: frozenset[str] = frozenset()


NO_EXCLUSIONS = Exclusions()
