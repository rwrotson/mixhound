from mixhound.models.domain import DiscoveryListeningLink, DiscoveryTrack

_PRIORITY = ("spotify", "bandcamp", "youtube", "soundcloud")


def _priority(service: str) -> int:
    """Same order as the client's servicePriority."""
    name = service.casefold()
    return next((index for index, key in enumerate(_PRIORITY) if key in name), len(_PRIORITY))


def playable_link(link: DiscoveryListeningLink) -> bool:
    """A link the integrated player can try: not known to be gone or unplayable."""
    if link.reachable is False or link.playback_available is False:
        return False
    name = link.service.casefold()
    return (link.target_type == "track" and any(key in name for key in _PRIORITY)) or (
        link.target_type == "album" and "bandcamp" in name
    )


def choose_preferred(track: DiscoveryTrack) -> None:
    """Keep a still-playable default source, otherwise default to the best available one."""
    candidates = [link for link in track.listening_links if playable_link(link)]
    if any(link.url == track.preferred_listening_url for link in candidates):
        return
    best = min(candidates, key=lambda link: _priority(link.service), default=None)
    track.preferred_listening_url = best.url if best else None
