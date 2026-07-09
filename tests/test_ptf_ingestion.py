from datetime import datetime
from zoneinfo import ZoneInfo

from app.main import app
from data_pipeline.epias.ptf_ingestion import (
    PtfIngestionService,
    split_date_range,
)
from data_pipeline.validation.time_series import find_missing_hourly_timestamps

ISTANBUL = ZoneInfo("Europe/Istanbul")


def test_parse_ptf_response_supports_nested_mcp_shape() -> None:
    service = PtfIngestionService(client=object())  # type: ignore[arg-type]
    response = {
        "body": {
            "items": [
                {
                    "date": "2024-01-01T00:00:00+03:00",
                    "hour": "00:00",
                    "price": 1750.25,
                    "priceUsd": 58.10,
                    "priceEur": 53.20,
                },
                {
                    "date": "2024-01-01",
                    "hour": "01:00",
                    "mcp": "1.800,50",
                    "priceUSD": "59.20",
                    "priceEUR": "54.10",
                },
            ]
        }
    }

    records = service.parse_ptf_response(response)

    assert len(records) == 2
    assert records[0]["timestamp"].isoformat() == "2024-01-01T00:00:00+03:00"
    assert records[0]["ptf_tl"] == 1750.25
    assert records[1]["timestamp"].isoformat() == "2024-01-01T01:00:00+03:00"
    assert records[1]["ptf_tl"] == 1800.50
    assert records[1]["source"] == "epias"


def test_split_date_range_uses_inclusive_calendar_chunks() -> None:
    chunks = split_date_range(
        datetime(2024, 1, 1, tzinfo=ISTANBUL),
        datetime(2024, 1, 31, 23, 59, 59, tzinfo=ISTANBUL),
        chunk_days=30,
    )

    assert len(chunks) == 2
    assert chunks[0][0].isoformat() == "2024-01-01T00:00:00+03:00"
    assert chunks[0][1].isoformat() == "2024-01-30T23:59:59+03:00"
    assert chunks[1][0].isoformat() == "2024-01-31T00:00:00+03:00"
    assert chunks[1][1].isoformat() == "2024-01-31T23:59:59+03:00"


def test_find_missing_hourly_timestamps() -> None:
    start = datetime(2024, 1, 1, 0, tzinfo=ISTANBUL)
    end = datetime(2024, 1, 1, 3, tzinfo=ISTANBUL)
    existing = [
        datetime(2024, 1, 1, 0, tzinfo=ISTANBUL),
        datetime(2024, 1, 1, 2, tzinfo=ISTANBUL),
        datetime(2024, 1, 1, 3, tzinfo=ISTANBUL),
    ]

    missing = find_missing_hourly_timestamps(start, end, existing)

    assert missing == [datetime(2024, 1, 1, 1, tzinfo=ISTANBUL)]


def test_ptf_routes_are_registered_in_openapi() -> None:
    paths = app.openapi()["paths"]

    assert "get" in paths["/api/epias/ptf/status"]
    assert "post" in paths["/api/epias/ptf/ingest"]

