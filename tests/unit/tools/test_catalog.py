from mixhound.services.ports import CatalogRecording
from mixhound.tools.catalog import describe_recording
from mixhound.tools.deps import MAX_DESCRIBED
from tests.fakes import FakeCatalog
from tests.support.agents import context, tool_deps

WORDS = CatalogRecording(artist="Low", title="Words", confirmed_by=("musicbrainz",))


async def test_describe_asks_the_candidate_profile() -> None:
    catalog = FakeCatalog({("Low", "Words"): WORDS})
    deps = tool_deps(catalog=catalog)

    text = await describe_recording(context(deps), "Low", "Words", "live")

    assert text.startswith("Recording: Low — Words")
    assert catalog.calls == [("candidate", "Low", "Words", "live", None)]


async def test_describe_reports_unknown_and_unavailable() -> None:
    unknown = await describe_recording(context(tool_deps()), "Low", "Nope")
    down = await describe_recording(context(tool_deps(catalog=FakeCatalog(down=True))), "A", "B")

    assert unknown == "No catalog knows Low — Nope."
    assert down == "The music catalogs are unavailable right now."


async def test_describe_stops_at_its_cap() -> None:
    catalog = FakeCatalog()
    deps = tool_deps(catalog=catalog)
    for _ in range(MAX_DESCRIBED):
        await describe_recording(context(deps), "A", "B")

    assert "limit reached" in await describe_recording(context(deps), "A", "B")
    assert len(catalog.calls) == MAX_DESCRIBED
