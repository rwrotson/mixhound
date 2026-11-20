from mixhound.config import PACKAGED_SKILLS_DIR
from mixhound.infrastructure.cache import MemoryCache
from mixhound.infrastructure.skills import load_catalog
from mixhound.memory.context import RunHistory
from mixhound.models.domain import (
    DiscoveryReference,
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySettings,
)
from mixhound.services.agentic import AgenticWorkspace, RunServices
from mixhound.services.audio import AudioAnalyzer, AudioMatcher
from mixhound.services.exclusions import Exclusions
from mixhound.services.pages import PageInspector
from mixhound.services.ports import CatalogLead, CatalogLink, CatalogPage, CatalogRecording
from mixhound.services.research import ResearchSteps
from mixhound.services.run_state import RunState
from mixhound.services.seed import Seeder
from mixhound.services.verification import ResultVerifier
from tests.fakes import FakeAgents, FakeCatalog, FakeFetch, Outcome

SKILLS = load_catalog(PACKAGED_SKILLS_DIR)
HISTORY = RunHistory(runs=[], tracks=[])
REVIEW = "https://reviews.example/duster-echo"


def reference(artist: str, title: str, **changes: object) -> DiscoveryReference:
    return DiscoveryReference.model_validate(
        {"kind": "manual", "artist": artist, "title": title, **changes}
    )


def leads(*artists: str) -> tuple[CatalogLead, ...]:
    return tuple(
        CatalogLead("lastfm_similar_tracks", artist, "Song", "Last.fm", "similar track")
        for artist in artists
    )


LOW = reference("Low", "Words", priority=5)
CODEINE = reference("Codeine", "D", priority=3)
SWANS = reference("Swans", "Loud", polarity="negative")
RECORDINGS = {
    ("Low", "Words"): CatalogRecording(
        "Low",
        "Words",
        mbid="r1",
        tags={"musicbrainz": ("slowcore", "sadcore", "dream pop")},
        leads=leads("Duster", "Bedhead", "Idaho", "Spain"),
    ),
    ("Codeine", "D"): CatalogRecording("Codeine", "D", leads=leads("Duster", "Seam", "Rex")),
    ("Swans", "Loud"): CatalogRecording("Swans", "Loud", leads=leads("Gira")),
    ("Duster", "Echo"): CatalogRecording(
        "Duster",
        "Echo",
        confirmed_by=("musicbrainz",),
        mbid="d1",
        year=1998,
        pages=(CatalogPage("musicbrainz", "https://musicbrainz.org/recording/d1"),),
        links=(CatalogLink("spotify", "https://open.spotify.com/track/abc", "track"),),
    ),
}


def workspace(
    *,
    catalog: FakeCatalog | None = None,
    outcomes: dict[str, Outcome] | None = None,
    audio: AudioAnalyzer | None = None,
    cancelled: bool = False,
    **settings: object,
) -> tuple[AgenticWorkspace, FakeCatalog, FakeAgents]:
    catalog = catalog or FakeCatalog(dict(RECORDINGS))
    agents = FakeAgents(outcomes or {})
    pages = FakeFetch({"reviews.example/duster-echo": b"<html>Bedhead - Song, hushed</html>"})
    verifier = ResultVerifier(catalog, PageInspector(pages), MemoryCache())
    run = DiscoveryRun(
        session_id=SWANS.id,
        input=DiscoveryRunInput(
            references=[CODEINE, LOW, SWANS],
            settings=DiscoverySettings.model_validate({"depth": "quick", **settings}),
        ),
    )
    state = RunState(run, cancelled=lambda: cancelled)
    services = RunServices(
        Seeder(catalog, audio),
        verifier,
        AudioMatcher(None),
        ResearchSteps(agents, verifier, SKILLS),
    )
    return AgenticWorkspace(state, HISTORY, Exclusions(), services), catalog, agents
