from urllib.parse import parse_qs

import httpx
import pytest

from app.core.config import Settings
from data_pipeline.epias.client import (
    EpiasClient,
    EpiasCredentialsError,
)


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "epias_base_url": "https://seffaflik.example",
        "epias_auth_url": "https://giris.example",
        "epias_username": None,
        "epias_password": None,
        "epias_request_timeout": 5,
        "epias_max_retries": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_config_loads_epias_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EPIAS_BASE_URL", "https://custom.example")
    monkeypatch.setenv("EPIAS_REQUEST_TIMEOUT", "12")
    monkeypatch.setenv("EPIAS_MAX_RETRIES", "2")
    monkeypatch.setenv("EPIAS_PTF_ENDPOINT", "/custom/mcp")

    config = Settings(_env_file=None)

    assert config.epias_base_url == "https://custom.example"
    assert config.epias_request_timeout == 12
    assert config.epias_max_retries == 2
    assert config.epias_ptf_endpoint == "/custom/mcp"


def test_client_initializes_without_credentials() -> None:
    client = EpiasClient(config=make_settings())

    assert client.credentials_configured is False
    with pytest.raises(EpiasCredentialsError):
        client.authenticate()
    client.close()


def test_authenticated_post_uses_and_caches_tgt() -> None:
    auth_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal auth_calls
        if request.url.path == "/cas/v1/tickets":
            auth_calls += 1
            form = parse_qs(request.content.decode())
            assert form == {
                "username": ["user@example.com"],
                "password": ["secret"],
            }
            return httpx.Response(201, text="TGT-test-token")

        assert request.headers["TGT"] == "TGT-test-token"
        return httpx.Response(200, json={"items": [{"value": 1}]})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = EpiasClient(
        config=make_settings(
            epias_username="user@example.com",
            epias_password="secret",
        ),
        http_client=http_client,
    )

    first = client.post("/electricity-service/v1/test", {"startDate": "2026-01-01"})
    second = client.post("/electricity-service/v1/test", {})

    assert first.status_code == 200
    assert first.data["items"][0]["value"] == 1
    assert second.status_code == 200
    assert auth_calls == 1
    http_client.close()


def test_post_retries_retryable_response() -> None:
    request_count = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(503, json={"message": "temporary"})
        return httpx.Response(200, json={"status": "ok"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = EpiasClient(
        config=make_settings(epias_max_retries=1),
        http_client=http_client,
        sleep=delays.append,
    )

    response = client.post("/electricity-service/v1/test", {}, use_auth=False)

    assert response.data == {"status": "ok"}
    assert request_count == 2
    assert delays == [0.5]
    http_client.close()
