from pydantic_ai import RunContext

from mixhound.services.dossier import describe_recording as describe
from mixhound.services.ports import CatalogUnavailableError
from mixhound.tools.deps import ToolDeps


async def describe_recording(
    ctx: RunContext[ToolDeps], artist: str, title: str, version: str | None = None
) -> str:
    """Look a recording up in the music catalogs: identity, tags, credits, texts, relations.

    Each fact names its source; a recording no catalog knows may not exist as named.

    Args:
        artist: The credited artist.
        title: The recording title without version words.
        version: A version qualifier such as "live" or "remix", if any.
    """
    deps = ctx.deps
    if not deps.take("describe_recording"):
        return "describe_recording limit reached for this run."
    try:
        recording = await deps.toolbox.catalog.candidate(artist, title, version=version)
    except CatalogUnavailableError:
        return "The music catalogs are unavailable right now."
    if recording is None:
        return f"No catalog knows {artist} — {title}."
    return describe(recording)
