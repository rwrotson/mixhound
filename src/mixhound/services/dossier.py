from mixhound.services.ports import CatalogRecording

MAX_FACTS = 40
MAX_TAGS = 8
MAX_LINKS = 8


def describe_recording(recording: CatalogRecording) -> str:
    """Render what the catalogs know about a recording as compact lines with their sources."""
    lines = [f"Recording: {recording.artist} — {recording.title}"]
    identity = [
        f"year {recording.year}" if recording.year else "",
        f"MusicBrainz {recording.mbid}" if recording.mbid else "",
        f"ISRC {', '.join(recording.isrcs)}" if recording.isrcs else "",
    ]
    if any(identity):
        lines.append("Identity: " + "; ".join(part for part in identity if part))
    lines.append(
        "Confirmed by: " + (", ".join(recording.confirmed_by) or "no catalog (unconfirmed)")
    )
    lines.extend(
        f"Tags ({source}): {', '.join(tags[:MAX_TAGS])}"
        for source, tags in recording.tags.items()
        if tags
    )
    for fact in recording.facts[:MAX_FACTS]:
        where = f" [{fact.source}{' ' + fact.url if fact.url else ''}]" if fact.source else ""
        lines.append(f"- {fact.kind}: {fact.text}{where}")
    lines.extend(f"Page ({page.source}): {page.url}" for page in recording.pages[:MAX_LINKS])
    return "\n".join(lines)
