from fastapi.testclient import TestClient

from tests.fakes import FakeFetch
from tests.support.runs import Harness, api

PREFIX = "/api/v1"


def test_providers_report_the_configuration() -> None:
    harness = Harness()

    with TestClient(api(harness)) as client:
        response = client.get(f"{PREFIX}/providers")

    assert response.json() == harness.providers.model_dump()


def test_resolve_url() -> None:
    page = b'<meta property="og:title" content="Song">'
    harness = Harness(fetch=FakeFetch({"example.org/a": page}))

    with TestClient(api(harness)) as client:
        resolved = client.post(f"{PREFIX}/resolve-url", json={"url": "https://example.org/a"})
        missing = client.post(f"{PREFIX}/resolve-url", json={"url": "https://example.org/b"})

    assert resolved.json() == {
        "url": "https://example.org/a",
        "service": "Other",
        "artist": None,
        "title": "Song",
    }
    assert missing.status_code == 422


def test_ready_needs_the_ownership_of_the_runs() -> None:
    with TestClient(api(Harness(owner=False))) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
