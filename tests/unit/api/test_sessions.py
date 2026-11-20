from uuid import uuid4

from fastapi.testclient import TestClient

from tests.support.runs import Harness, api

PREFIX = "/api/v1"
REFERENCE = {"kind": "manual", "artist": "Artist", "title": "Title"}


def test_session_lifecycle() -> None:
    with TestClient(api(Harness())) as client:
        created = client.post(f"{PREFIX}/sessions", json={"name": "Test"})
        session_id = created.json()["id"]
        renamed = client.patch(f"{PREFIX}/sessions/{session_id}", json={"name": "Other"})
        added = client.post(f"{PREFIX}/sessions/{session_id}/references", json=REFERENCE)
        reference_id = added.json()["references"][0]["id"]
        updated = client.patch(
            f"{PREFIX}/sessions/{session_id}/references/{reference_id}",
            json={**REFERENCE, "priority": 5},
        )
        manual = client.post(
            f"{PREFIX}/sessions/{session_id}/references/manual",
            json={"lines": "Band — Song", "polarity": "negative"},
        )
        deleted = client.delete(f"{PREFIX}/sessions/{session_id}/references/{reference_id}")
        listed = client.get(f"{PREFIX}/sessions")
        read = client.get(f"{PREFIX}/sessions/{session_id}")

    assert created.status_code == 201
    assert "version" not in created.json()
    assert renamed.json()["name"] == "Other"
    assert added.status_code == 201
    assert updated.json()["references"][0]["priority"] == 5
    assert manual.status_code == 201
    assert [item["polarity"] for item in manual.json()["references"]] == ["positive", "negative"]
    assert [item["artist"] for item in deleted.json()["references"]] == ["Band"]
    assert [item["id"] for item in listed.json()] == [session_id]
    assert read.json()["references"] == deleted.json()["references"]


def test_session_errors_are_problem_details() -> None:
    with TestClient(api(Harness())) as client:
        session_id = client.post(f"{PREFIX}/sessions", json={"name": "Test"}).json()["id"]
        missing = client.get(f"{PREFIX}/sessions/{uuid4()}")
        malformed = client.post(
            f"{PREFIX}/sessions/{session_id}/references/manual", json={"lines": "nothing"}
        )
        no_url = client.post(
            f"{PREFIX}/sessions/{session_id}/references", json={**REFERENCE, "kind": "url"}
        )
        unknown_reference = client.delete(f"{PREFIX}/sessions/{session_id}/references/{uuid4()}")
        empty_name = client.post(f"{PREFIX}/sessions", json={"name": ""})

    assert missing.status_code == 404
    assert missing.json()["title"] == "Not Found"
    assert malformed.status_code == 422
    assert malformed.json()["detail"] == "Line 1 must be Artist — Track"
    assert no_url.status_code == 422
    assert unknown_reference.status_code == 404
    assert empty_name.status_code == 422


def test_default_settings_seed_new_sessions() -> None:
    with TestClient(api(Harness())) as client:
        saved = client.patch(f"{PREFIX}/settings", json={"mode": "agentic", "minutes": 5})
        read = client.get(f"{PREFIX}/settings")
        session = client.post(f"{PREFIX}/sessions", json={"name": "Test"}).json()
        invalid = client.patch(f"{PREFIX}/settings", json={"year_from": 2000, "year_to": 1990})

    assert saved.json() == read.json()
    assert (read.json()["mode"], read.json()["minutes"]) == ("agentic", 5)
    assert "version" not in read.json()
    assert session["settings"] == read.json()
    assert invalid.status_code == 422
