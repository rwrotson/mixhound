import json
from uuid import uuid4

from fastapi.testclient import TestClient

from mixhound.models.domain import DiscoveryRun, DiscoveryRunInput, DiscoveryTrack
from tests.support.runs import Harness, api

PREFIX = "/api/v1"
REFERENCE = {"kind": "manual", "artist": "Artist", "title": "Title"}


def run_request(**values: object) -> dict[str, object]:
    return {"instructions": "More", "references": [REFERENCE], "settings": {}, **values}


def sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_run_is_created_followed_and_read() -> None:
    harness = Harness()
    with TestClient(api(harness)) as client:
        session = client.post(f"{PREFIX}/sessions", json={"name": "Test"}).json()

        created = client.post(f"{PREFIX}/sessions/{session['id']}/runs", json=run_request())
        run_id = created.json()["id"]
        with client.stream("GET", f"{PREFIX}/runs/{run_id}/events") as stream:
            assert stream.headers["content-type"].startswith("text/event-stream")
            events = sse(stream.read().decode())
        run = client.get(f"{PREFIX}/runs/{run_id}")
        progress = client.get(f"{PREFIX}/runs/{run_id}/progress")

    assert created.status_code == 202
    assert created.json()["status"] == "queued"
    assert "version" not in created.json()
    assert [(kind, data["status"]) for kind, data in events if kind == "status"] == [
        ("status", "queued"),
        ("status", "running"),
        ("status", "completed"),
    ]
    assert run.json()["status"] == "completed"
    assert progress.status_code == 200
    assert progress.json() is None


def test_run_errors_are_problem_details() -> None:
    harness = Harness()
    with TestClient(api(harness)) as client:
        session = client.post(f"{PREFIX}/sessions", json={"name": "Test"}).json()
        invalid = client.post(
            f"{PREFIX}/sessions/{session['id']}/runs", json=run_request(instructions=" ")
        )
        unknown = client.get(f"{PREFIX}/runs/{uuid4()}")
        unknown_events = client.get(f"{PREFIX}/runs/{uuid4()}/events")
        unknown_progress = client.get(f"{PREFIX}/runs/{uuid4()}/progress")
        missing_session = client.post(f"{PREFIX}/sessions/{uuid4()}/runs", json=run_request())

    assert invalid.status_code == 422
    assert invalid.headers["content-type"] == "application/problem+json"
    assert invalid.json()["detail"] == "A prompt and a positive reference are required"
    assert unknown.status_code == 404
    assert unknown.json()["detail"] == "Run not found"
    assert unknown_events.status_code == 404
    assert unknown_progress.status_code == 404
    assert missing_session.status_code == 404


def test_cancel_of_a_finished_run_conflicts() -> None:
    harness = Harness()
    session_id = uuid4()
    finished = DiscoveryRun(session_id=session_id, input=DiscoveryRunInput(), status="completed")
    harness.runs.runs[finished.id] = finished

    with TestClient(api(harness)) as client:
        response = client.post(f"{PREFIX}/runs/{finished.id}/cancel")

    assert response.status_code == 409
    assert response.json()["detail"] == "Run is not active"


def test_feedback_on_a_result() -> None:
    harness = Harness()
    track = DiscoveryTrack(artist="A", title="B", audio_features={"tempo": 1.0})
    with TestClient(api(harness)) as client:
        session = client.post(f"{PREFIX}/sessions", json={"name": "Test"}).json()
        run = DiscoveryRun(
            session_id=session["id"], input=DiscoveryRunInput(), status="completed", results=[track]
        )
        harness.runs.runs[run.id] = run
        harness.sessions.sessions[run.session_id].run_ids.append(run.id)
        url = f"{PREFIX}/runs/{run.id}/tracks/{track.id}/feedback"

        marked = client.post(url, json={"action": "mark", "polarity": "negative", "weight": 2})
        incomplete = client.post(url, json={"action": "mark", "polarity": "positive"})
        unknown = client.post(
            f"{PREFIX}/runs/{run.id}/tracks/{uuid4()}/feedback", json={"action": "skip"}
        )
        read = client.get(f"{PREFIX}/runs/{run.id}").json()

    assert marked.json() == {
        "weight": 2,
        "polarity": "negative",
        "skipped": False,
        "text": None,
        "tags": [],
        "excluded": False,
        "avoid_similar": False,
    }
    assert incomplete.status_code == 422
    assert unknown.status_code == 404
    assert "audio_features" not in read["results"][0]
