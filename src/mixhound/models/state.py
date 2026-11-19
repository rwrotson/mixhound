from pydantic import BaseModel


class RunProgress(BaseModel):
    """Where an active run is now; kept in memory only, never stored."""

    stage: str
    # The stage's place among the run's stages, 1-based.
    step: int
    steps: int
    # Research only: seconds since the agent started, and its time limit.
    elapsed: float = 0.0
    timeout: float = 0.0
    searches: int = 0
    # Candidates the research has written to its interim notes so far.
    candidates: int = 0
