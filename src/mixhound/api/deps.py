from typing import Annotated, Protocol

from fastapi import Depends, Request

from mixhound.models.responses import ProvidersOut
from mixhound.services.runs import RunService
from mixhound.services.sessions import SessionService


class Services(Protocol):
    """What the routes use from the application resources."""

    @property
    def sessions(self) -> SessionService:
        """Sessions, references, feedback and default settings."""
        ...

    @property
    def runs(self) -> RunService:
        """Run creation, queueing, cancellation and progress."""
        ...

    @property
    def providers(self) -> ProvidersOut:
        """What the process is configured to use."""
        ...


def services(request: Request) -> Services:
    """Return the resources the lifespan opened."""
    resources: Services = request.state.container
    return resources


def sessions(request: Request) -> SessionService:
    """Return the session service."""
    return services(request).sessions


def runs(request: Request) -> RunService:
    """Return the run service."""
    return services(request).runs


ServicesDep = Annotated[Services, Depends(services)]
SessionsDep = Annotated[SessionService, Depends(sessions)]
RunsDep = Annotated[RunService, Depends(runs)]
