import pytest
from fastapi.testclient import TestClient

from mixhound.config import ApiSettings, Settings
from mixhound.main import create_app, serve


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


def test_serve_runs_the_application_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **options: calls.append({"app": app, **options}))

    serve()

    assert calls == [
        {"app": create_app, "factory": True, "host": "127.0.0.1", "port": 8000, "access_log": False}
    ]


def test_validation_errors_are_problem_details() -> None:
    application = create_app(Settings())

    @application.get("/items/{item_id}")
    async def item(item_id: int) -> dict[str, int]:
        return {"id": item_id}

    with TestClient(application) as client:
        response = client.get("/items/abc")

    assert response.status_code == 422
    assert response.json()["detail"] == "Request validation failed"
    assert response.json()["errors"][0]["loc"] == ["path", "item_id"]
