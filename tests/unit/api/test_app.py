from fastapi.testclient import TestClient

from mixhound.config import ApiSettings, Settings
from mixhound.main import create_app


def test_liveness_and_readiness() -> None:
    with TestClient(create_app(Settings())) as client:
        assert client.get("/live").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ok"}


def test_unknown_route_is_problem_details() -> None:
    with TestClient(create_app(Settings())) as client:
        response = client.get("/missing")

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "about:blank",
        "title": "Not Found",
        "status": 404,
        "detail": "Not Found",
        "instance": "/missing",
    }


def test_docs_can_be_disabled() -> None:
    settings = Settings(api=ApiSettings(docs_enabled=False))

    with TestClient(create_app(settings)) as client:
        assert client.get("/openapi.json").status_code == 404


def test_cors_origins_are_allowed() -> None:
    settings = Settings(api=ApiSettings(cors_origins=["http://localhost:5173"]))

    with TestClient(create_app(settings)) as client:
        response = client.get("/live", headers={"Origin": "http://localhost:5173"})

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
