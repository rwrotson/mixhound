from uuid import UUID

from mixhound.memory.repository import RunRepository, SessionRepository
from mixhound.models.domain import (
    DiscoveryReference,
    DiscoveryRun,
    DiscoverySession,
    DiscoverySettings,
    FeedbackEvent,
    FeedbackState,
    Polarity,
    now_utc,
)
from mixhound.services.errors import InvalidInputError, NotFoundError
from mixhound.services.identity import parse_manual_references
from mixhound.services.pages import PageInspector, PageRecording
from mixhound.services.ports import FetchError


class SessionNotFoundError(NotFoundError):
    """A session does not exist."""


class RunNotFoundError(NotFoundError):
    """A run does not exist."""


class TrackNotFoundError(NotFoundError):
    """A run has no result or lead with the requested ID."""


class SessionReferenceNotFoundError(NotFoundError):
    """A reference ID is absent from the session."""


class InvalidSessionReferenceError(InvalidInputError):
    """A session reference is malformed or points outside the session."""


class InvalidFeedbackError(InvalidInputError):
    """A feedback action lacks the fields it needs."""


class URLResolveError(InvalidInputError):
    """A public URL does not describe a recording that could be read."""


async def require_session(sessions: SessionRepository, session_id: UUID) -> DiscoverySession:
    """Return a session or raise SessionNotFoundError."""
    session = await sessions.get(session_id)
    if session is None:
        raise SessionNotFoundError("Session not found")
    return session


async def require_run(runs: RunRepository, run_id: UUID) -> DiscoveryRun:
    """Return a run or raise RunNotFoundError."""
    run = await runs.get(run_id)
    if run is None:
        raise RunNotFoundError("Run not found")
    return run


class SessionService:
    """Sessions, their references and feedback, and the defaults of new sessions."""

    def __init__(
        self, sessions: SessionRepository, runs: RunRepository, inspector: PageInspector
    ) -> None:
        self.sessions = sessions
        self.runs = runs
        self.inspector = inspector

    async def list_all(self) -> list[DiscoverySession]:
        """Return every session, oldest first."""
        return await self.sessions.list()

    async def get(self, session_id: UUID) -> DiscoverySession:
        """Return one session."""
        return await require_session(self.sessions, session_id)

    async def create(
        self,
        name: str,
        instructions: str,
        references: list[DiscoveryReference],
        settings: DiscoverySettings | None,
    ) -> DiscoverySession:
        """Create a session; without settings it starts from the defaults."""
        session = DiscoverySession(
            name=name,
            instructions=instructions,
            references=references,
            settings=settings if settings is not None else await self.sessions.default_settings(),
        )
        await self.sessions.save(session)
        return session

    async def rename(self, session_id: UUID, name: str) -> DiscoverySession:
        """Rename a session."""
        session = await self.get(session_id)
        session.name = name
        return await self._save(session)

    async def default_settings(self) -> DiscoverySettings:
        """Return the settings new sessions start from."""
        return await self.sessions.default_settings()

    async def save_default_settings(self, settings: DiscoverySettings) -> DiscoverySettings:
        """Replace the settings new sessions start from."""
        await self.sessions.save_default_settings(settings)
        return settings

    async def add_reference(
        self, session_id: UUID, reference: DiscoveryReference
    ) -> DiscoverySession:
        """Validate and append one reference."""
        session = await self._checked_reference(session_id, reference)
        session.references.append(reference)
        return await self._save(session)

    async def add_manual(
        self, session_id: UUID, lines: str, polarity: Polarity
    ) -> DiscoverySession:
        """Parse typed `Artist — Track` lines and append their references."""
        session = await self.get(session_id)
        try:
            parsed = parse_manual_references(lines)
        except ValueError as error:
            raise InvalidSessionReferenceError(str(error)) from error
        session.references.extend(
            DiscoveryReference(kind="manual", artist=artist, title=title, polarity=polarity)
            for artist, title in parsed
        )
        return await self._save(session)

    async def update_reference(
        self, session_id: UUID, reference: DiscoveryReference
    ) -> DiscoverySession:
        """Replace an existing reference after checking its target."""
        session = await self._checked_reference(session_id, reference)
        for index, item in enumerate(session.references):
            if item.id == reference.id:
                session.references[index] = reference
                return await self._save(session)
        raise SessionReferenceNotFoundError("Reference not found")

    async def delete_reference(self, session_id: UUID, reference_id: UUID) -> DiscoverySession:
        """Delete one reference."""
        session = await self.get(session_id)
        before = len(session.references)
        session.references = [item for item in session.references if item.id != reference_id]
        if len(session.references) == before:
            raise SessionReferenceNotFoundError("Reference not found")
        return await self._save(session)

    async def record_feedback(
        self, run_id: UUID, track_id: UUID, fields: dict[str, object]
    ) -> FeedbackState:
        """Validate and record a feedback action on a result; return its new state."""
        if fields.get("action") == "mark" and (
            fields.get("polarity") is None
            or (fields.get("polarity") != "neutral" and fields.get("weight") is None)
        ):
            raise InvalidFeedbackError("A polarity, and a weight unless neutral, are required")
        run = await require_run(self.runs, run_id)
        session = await self.sessions.get(run.session_id)
        if (
            session is None
            or run.id not in session.run_ids
            or all(track.id != track_id for track in run.results + run.unverified_leads)
        ):
            raise TrackNotFoundError("Track not found")
        event = FeedbackEvent.model_validate({"run_id": run.id, "track_id": track_id, **fields})
        session.feedback_events.append(event)
        await self._save(session)
        return session.feedback_for(track_id)

    async def resolve_url(self, url: str) -> PageRecording:
        """Read the recording a public URL describes."""
        try:
            return await self.inspector.resolve(url)
        except FetchError as error:
            raise URLResolveError(f"The page could not be read: {error}") from error

    async def _checked_reference(
        self, session_id: UUID, reference: DiscoveryReference
    ) -> DiscoverySession:
        """Load the session and check that the reference's target exists."""
        session = await self.get(session_id)
        if reference.kind == "previous_result" and not await self._has_result(
            session, reference.previous_track_id
        ):
            raise InvalidSessionReferenceError("Previous result not found")
        if reference.kind == "url" and not reference.url:
            raise InvalidSessionReferenceError("URL is required")
        return session

    async def _has_result(self, session: DiscoverySession, track_id: UUID | None) -> bool:
        """Whether a run of the session published the track."""
        if track_id is None:
            return False
        for run_id in session.run_ids:
            run = await self.runs.get(run_id)
            if run is not None and any(track.id == track_id for track in run.results):
                return True
        return False

    async def _save(self, session: DiscoverySession) -> DiscoverySession:
        """Store a changed session with a fresh update time."""
        session.updated_at = now_utc()
        await self.sessions.save(session)
        return session
