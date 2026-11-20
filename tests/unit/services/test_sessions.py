from uuid import uuid4

import pytest

from mixhound.models.domain import (
    DiscoveryReference,
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySettings,
    DiscoveryTrack,
)
from mixhound.services.pages import PageRecording
from mixhound.services.sessions import (
    InvalidFeedbackError,
    InvalidSessionReferenceError,
    RunNotFoundError,
    SessionNotFoundError,
    SessionReferenceNotFoundError,
    TrackNotFoundError,
    URLResolveError,
)
from tests.fakes import FakeFetch
from tests.support.runs import Harness, positive


async def test_new_session_starts_from_the_default_settings() -> None:
    harness = Harness()
    defaults = DiscoverySettings(mode="agentic", minutes=5)
    sessions = harness.session_service
    assert await sessions.save_default_settings(defaults) == defaults

    plain = await sessions.create("Plain", "", [], None)
    custom = await sessions.create("Custom", "More", [positive()], DiscoverySettings())

    assert plain.settings == defaults
    assert await sessions.default_settings() == defaults
    assert custom.settings == DiscoverySettings()
    assert [item.name for item in await sessions.list_all()] == ["Plain", "Custom"]


async def test_rename_and_missing_session() -> None:
    harness = Harness()
    session = await harness.session()

    renamed = await harness.session_service.rename(session.id, "Other")

    assert renamed.name == "Other"
    assert renamed.updated_at > session.updated_at
    assert (await harness.session_service.get(session.id)).name == "Other"
    with pytest.raises(SessionNotFoundError, match="Session not found"):
        await harness.session_service.get(uuid4())


async def test_references_are_added_replaced_and_deleted() -> None:
    harness = Harness()
    session = await harness.session()
    sessions = harness.session_service
    reference = positive()

    await sessions.add_reference(session.id, reference)
    changed = reference.model_copy(update={"priority": 5})
    updated = await sessions.update_reference(session.id, changed)
    assert [item.priority for item in updated.references] == [5]

    emptied = await sessions.delete_reference(session.id, reference.id)
    assert emptied.references == []
    with pytest.raises(SessionReferenceNotFoundError):
        await sessions.delete_reference(session.id, reference.id)
    with pytest.raises(SessionReferenceNotFoundError):
        await sessions.update_reference(session.id, changed)


async def test_previous_result_must_be_a_result_of_the_session() -> None:
    harness = Harness()
    session = await harness.session()
    published, lead = DiscoveryTrack(artist="A", title="B"), DiscoveryTrack(artist="C", title="D")
    run = DiscoveryRun(
        session_id=session.id,
        input=DiscoveryRunInput(),
        status="completed",
        results=[published],
        unverified_leads=[lead],
    )
    await harness.runs.save(run)
    session.run_ids = [uuid4(), run.id]
    await harness.sessions.save(session)

    def previous(track_id: object) -> DiscoveryReference:
        return DiscoveryReference.model_validate(
            {"kind": "previous_result", "artist": "A", "title": "B", "previous_track_id": track_id}
        )

    added = await harness.session_service.add_reference(session.id, previous(published.id))

    assert len(added.references) == 1
    for track_id in (lead.id, None):
        with pytest.raises(InvalidSessionReferenceError, match="Previous result not found"):
            await harness.session_service.add_reference(session.id, previous(track_id))


async def test_url_reference_needs_a_url() -> None:
    harness = Harness()
    session = await harness.session()

    with pytest.raises(InvalidSessionReferenceError, match="URL is required"):
        await harness.session_service.add_reference(
            session.id, DiscoveryReference(kind="url", artist="A", title="B")
        )


async def test_manual_lines_become_references() -> None:
    harness = Harness()
    session = await harness.session()

    updated = await harness.session_service.add_manual(
        session.id, "Artist — One\n\nOther - Two", "negative"
    )

    assert [(item.artist, item.title, item.polarity) for item in updated.references] == [
        ("Artist", "One", "negative"),
        ("Other", "Two", "negative"),
    ]
    with pytest.raises(InvalidSessionReferenceError, match="Line 1"):
        await harness.session_service.add_manual(session.id, "no separator", "positive")


async def test_feedback_is_recorded_on_results_and_leads() -> None:
    harness = Harness()
    session = await harness.session()
    result, lead = DiscoveryTrack(artist="A", title="B"), DiscoveryTrack(artist="C", title="D")
    run = DiscoveryRun(
        session_id=session.id,
        input=DiscoveryRunInput(),
        status="completed",
        results=[result],
        unverified_leads=[lead],
    )
    await harness.runs.save(run)
    session.run_ids = [run.id]
    await harness.sessions.save(session)
    sessions = harness.session_service

    marked = await sessions.record_feedback(
        run.id, result.id, {"action": "mark", "polarity": "positive", "weight": 4}
    )
    skipped = await sessions.record_feedback(run.id, lead.id, {"action": "skip"})

    assert (marked.polarity, marked.weight) == ("positive", 4)
    assert skipped.skipped
    stored = await sessions.get(session.id)
    assert [event.track_id for event in stored.feedback_events] == [result.id, lead.id]


@pytest.mark.parametrize(
    "fields",
    [{"action": "mark", "weight": 3}, {"action": "mark", "polarity": "positive"}],
)
async def test_mark_needs_a_polarity_and_a_weight(fields: dict[str, object]) -> None:
    with pytest.raises(InvalidFeedbackError, match="polarity"):
        await Harness().session_service.record_feedback(uuid4(), uuid4(), fields)


async def test_feedback_needs_a_known_run_and_track() -> None:
    harness = Harness()
    session = await harness.session()
    linked = DiscoveryRun(session_id=session.id, input=DiscoveryRunInput())
    unlinked = DiscoveryRun(
        session_id=session.id,
        input=DiscoveryRunInput(),
        results=[track := DiscoveryTrack(artist="A", title="B")],
    )
    orphan = DiscoveryRun(session_id=uuid4(), input=DiscoveryRunInput(), results=[track])
    for run in (linked, unlinked, orphan):
        await harness.runs.save(run)
    session.run_ids = [linked.id]
    await harness.sessions.save(session)
    sessions = harness.session_service

    with pytest.raises(RunNotFoundError):
        await sessions.record_feedback(uuid4(), track.id, {"action": "skip"})
    for run in (linked, unlinked, orphan):
        with pytest.raises(TrackNotFoundError):
            await sessions.record_feedback(run.id, track.id, {"action": "skip"})


async def test_resolve_url_reads_the_page_or_explains_why_not() -> None:
    page = b'<meta property="og:title" content="Title">'
    harness = Harness(fetch=FakeFetch({"example.org/a": page}))

    resolved = await harness.session_service.resolve_url("https://example.org/a")

    assert resolved == PageRecording("https://example.org/a", "Other", title="Title")
    with pytest.raises(URLResolveError, match="could not be read: unreachable"):
        await harness.session_service.resolve_url("https://example.org/missing")
