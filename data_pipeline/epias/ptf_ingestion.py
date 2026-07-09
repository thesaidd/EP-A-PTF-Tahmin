import json
import logging
import math
import re
from collections.abc import Callable, Iterable
from datetime import date, datetime, time, timedelta
from typing import Any, TypedDict
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from data_pipeline.epias.client import (
    EpiasClient,
    EpiasCredentialsError,
    EpiasResponse,
)
from data_pipeline.epias.repository import save_raw_epias_response
from data_pipeline.validation.time_series import find_missing_hourly_timestamps

logger = logging.getLogger(__name__)

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")
DATE_KEYS = ("date", "timestamp", "dateTime", "datetime", "time")
TL_PRICE_KEYS = ("price", "mcp", "priceTl", "priceTL", "ptf", "ptfTl", "ptf_tl")
USD_PRICE_KEYS = ("priceUsd", "priceUSD", "ptfUsd", "ptf_usd")
EUR_PRICE_KEYS = ("priceEur", "priceEUR", "ptfEur", "ptf_eur")
WRAPPER_KEYS = ("items", "body", "data", "result")


class PtfRecord(TypedDict):
    timestamp: datetime
    ptf_tl: float
    ptf_usd: float | None
    ptf_eur: float | None
    source: str
    raw_record: dict[str, Any]


class PtfIngestionError(RuntimeError):
    """Raised for critical PTF ingestion configuration or persistence errors."""


def normalize_date_range(
    start_date: date | datetime,
    end_date: date | datetime,
) -> tuple[datetime, datetime]:
    start = _to_istanbul_datetime(start_date, is_end=False)
    end = _to_istanbul_datetime(end_date, is_end=True)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    return start, end


def split_date_range(
    start_date: date | datetime,
    end_date: date | datetime,
    chunk_days: int = 30,
) -> list[tuple[datetime, datetime]]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")

    start, end = normalize_date_range(start_date, end_date)
    chunks: list[tuple[datetime, datetime]] = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(
            chunk_start + timedelta(days=chunk_days) - timedelta(seconds=1),
            end,
        )
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(seconds=1)

    return chunks


class PtfIngestionService:
    def __init__(
        self,
        client: EpiasClient | None = None,
        endpoint: str | None = None,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self.client = client or EpiasClient()
        self.endpoint = endpoint or settings.epias_ptf_endpoint
        self.session_factory = session_factory

    def fetch_ptf_range(
        self,
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> EpiasResponse:
        payload = self.build_request_payload(start_date, end_date)
        return self.client.post(self.endpoint, payload, use_auth=True)

    def ingest_ptf_range(
        self,
        start_date: date | datetime,
        end_date: date | datetime,
        chunk_days: int = 30,
    ) -> dict[str, Any]:
        range_start, range_end = normalize_date_range(start_date, end_date)
        chunks = split_date_range(range_start, range_end, chunk_days)
        summary: dict[str, Any] = {
            "start_date": range_start.date().isoformat(),
            "end_date": range_end.date().isoformat(),
            "chunks_processed": 0,
            "records_fetched": 0,
            "records_inserted_or_updated": 0,
            "missing_hours": [],
            "errors": [],
        }

        for chunk_start, chunk_end in chunks:
            summary["chunks_processed"] += 1
            payload = self.build_request_payload(chunk_start, chunk_end)
            chunk_label = (
                f"{chunk_start.date().isoformat()}..{chunk_end.date().isoformat()}"
            )
            try:
                response = self.fetch_ptf_range(chunk_start, chunk_end)
                save_raw_epias_response(
                    endpoint_name="mcp",
                    endpoint_url=response.endpoint_url,
                    request_payload=payload,
                    response_json=response.data,
                    status_code=response.status_code,
                    data_start_date=chunk_start.date().isoformat(),
                    data_end_date=chunk_end.date().isoformat(),
                )
                records = self.parse_ptf_response(response.data)
                validation_errors = self.validate_ptf_records(records)
                if validation_errors:
                    summary["errors"].extend(
                        f"{chunk_label}: {message}" for message in validation_errors
                    )
                    continue

                summary["records_fetched"] += len(records)
                summary["records_inserted_or_updated"] += self.upsert_ptf_records(
                    records
                )
            except EpiasCredentialsError:
                raise
            except (PtfIngestionError, SQLAlchemyError, RuntimeError, ValueError) as exc:
                logger.exception("PTF ingestion chunk failed: chunk=%s", chunk_label)
                summary["errors"].append(f"{chunk_label}: {exc}")

        try:
            existing = self.get_existing_timestamps(range_start, range_end)
            summary["missing_hours"] = [
                timestamp.isoformat()
                for timestamp in find_missing_hourly_timestamps(
                    range_start,
                    range_end,
                    existing,
                )
            ]
        except SQLAlchemyError as exc:
            logger.exception("Could not calculate missing PTF hours.")
            summary["errors"].append(f"Missing-hour validation failed: {exc}")

        return summary

    def build_request_payload(
        self,
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> dict[str, str]:
        start, end = normalize_date_range(start_date, end_date)
        return {
            "startDate": start.isoformat(timespec="seconds"),
            "endDate": end.isoformat(timespec="seconds"),
        }

    def parse_ptf_response(self, response_json: Any) -> list[PtfRecord]:
        raw_records = _find_record_list(response_json)
        if raw_records is None:
            raise PtfIngestionError(
                "EPİAŞ PTF response does not contain a recognizable record list."
            )

        parsed_by_timestamp: dict[datetime, PtfRecord] = {}
        skipped = 0
        for index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, dict):
                skipped += 1
                continue
            try:
                timestamp = _parse_record_timestamp(raw_record)
                ptf_tl = _first_number(raw_record, TL_PRICE_KEYS, required=True)
                parsed_by_timestamp[timestamp] = {
                    "timestamp": timestamp,
                    "ptf_tl": ptf_tl,
                    "ptf_usd": _first_number(
                        raw_record,
                        USD_PRICE_KEYS,
                        required=False,
                    ),
                    "ptf_eur": _first_number(
                        raw_record,
                        EUR_PRICE_KEYS,
                        required=False,
                    ),
                    "source": "epias",
                    "raw_record": raw_record,
                }
            except (TypeError, ValueError) as exc:
                skipped += 1
                logger.warning("Skipping invalid PTF record index=%s: %s", index, exc)

        if skipped:
            logger.warning("Skipped %s invalid PTF record(s).", skipped)
        return sorted(parsed_by_timestamp.values(), key=lambda item: item["timestamp"])

    def validate_ptf_records(self, records: Iterable[PtfRecord]) -> list[str]:
        errors: list[str] = []
        seen: set[datetime] = set()
        record_count = 0

        for record in records:
            record_count += 1
            timestamp = record["timestamp"]
            if timestamp.tzinfo is None:
                errors.append(f"{timestamp!s}: timestamp is not timezone-aware")
            if timestamp in seen:
                errors.append(f"{timestamp.isoformat()}: duplicate timestamp")
            seen.add(timestamp)

            if not math.isfinite(record["ptf_tl"]):
                errors.append(f"{timestamp.isoformat()}: ptf_tl is not finite")
            for field in ("ptf_usd", "ptf_eur"):
                value = record[field]
                if value is not None and not math.isfinite(value):
                    errors.append(
                        f"{timestamp.isoformat()}: {field} is not finite"
                    )

        if record_count == 0:
            errors.append("No valid PTF records were returned")
        return errors

    def upsert_ptf_records(
        self,
        records: Iterable[PtfRecord],
        session: Session | None = None,
    ) -> int:
        record_list = list(records)
        if not record_list:
            return 0

        statement = text(
            """
            INSERT INTO ptf_hourly (
                "timestamp",
                ptf_tl,
                ptf_usd,
                ptf_eur,
                source,
                raw_record
            )
            VALUES (
                :timestamp,
                :ptf_tl,
                :ptf_usd,
                :ptf_eur,
                :source,
                CAST(:raw_record AS JSONB)
            )
            ON CONFLICT ("timestamp") DO UPDATE SET
                ptf_tl = EXCLUDED.ptf_tl,
                ptf_usd = EXCLUDED.ptf_usd,
                ptf_eur = EXCLUDED.ptf_eur,
                source = EXCLUDED.source,
                raw_record = EXCLUDED.raw_record,
                updated_at = NOW()
            """
        )
        values = [
            {
                "timestamp": record["timestamp"],
                "ptf_tl": record["ptf_tl"],
                "ptf_usd": record["ptf_usd"],
                "ptf_eur": record["ptf_eur"],
                "source": record["source"],
                "raw_record": json.dumps(record["raw_record"]),
            }
            for record in record_list
        ]
        owns_session = session is None
        database_session = session or self.session_factory()

        try:
            result = database_session.execute(statement, values)
            database_session.commit()
            return result.rowcount if result.rowcount >= 0 else len(record_list)
        except SQLAlchemyError:
            database_session.rollback()
            raise
        finally:
            if owns_session:
                database_session.close()

    def get_existing_timestamps(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> list[datetime]:
        with self.session_factory() as session:
            return list(
                session.scalars(
                    text(
                        """
                        SELECT "timestamp"
                        FROM ptf_hourly
                        WHERE "timestamp" >= :start_datetime
                          AND "timestamp" <= :end_datetime
                        ORDER BY "timestamp"
                        """
                    ),
                    {
                        "start_datetime": start_datetime,
                        "end_datetime": end_datetime,
                    },
                )
            )

    def get_status(self) -> dict[str, Any]:
        with self.session_factory() as session:
            row = session.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS total_rows,
                        MIN("timestamp") AS min_timestamp,
                        MAX("timestamp") AS max_timestamp,
                        MAX(updated_at) AS latest_updated_at
                    FROM ptf_hourly
                    """
                )
            ).mappings().one()
            return dict(row)


def _to_istanbul_datetime(value: date | datetime, is_end: bool) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=ISTANBUL_TIMEZONE)
        return value.astimezone(ISTANBUL_TIMEZONE)
    boundary = time(23, 59, 59) if is_end else time.min
    return datetime.combine(value, boundary, tzinfo=ISTANBUL_TIMEZONE)


def _find_record_list(value: Any, depth: int = 0) -> list[Any] | None:
    if depth > 5:
        return None
    if isinstance(value, list):
        if not value or any(
            isinstance(item, dict) and any(key in item for key in DATE_KEYS)
            for item in value
        ):
            return value
        return None
    if not isinstance(value, dict):
        return None

    for key in WRAPPER_KEYS:
        if key in value:
            result = _find_record_list(value[key], depth + 1)
            if result is not None:
                return result
    for nested_value in value.values():
        result = _find_record_list(nested_value, depth + 1)
        if result is not None:
            return result
    return None


def _parse_record_timestamp(record: dict[str, Any]) -> datetime:
    raw_date = next((record.get(key) for key in DATE_KEYS if record.get(key)), None)
    if raw_date is None:
        raise ValueError("missing date/timestamp field")

    if isinstance(raw_date, datetime):
        timestamp = raw_date
        date_has_time = True
    elif isinstance(raw_date, date):
        timestamp = datetime.combine(raw_date, time.min)
        date_has_time = False
    elif isinstance(raw_date, str):
        date_text = raw_date.strip().replace("Z", "+00:00")
        timestamp = datetime.fromisoformat(date_text)
        date_has_time = "T" in date_text or " " in date_text
    else:
        raise TypeError("unsupported date/timestamp value")

    if not date_has_time and record.get("hour") is not None:
        hour = _parse_hour(record["hour"])
        if hour == 24:
            timestamp += timedelta(days=1)
            hour = 0
        timestamp = timestamp.replace(hour=hour)

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=ISTANBUL_TIMEZONE)
    else:
        timestamp = timestamp.astimezone(ISTANBUL_TIMEZONE)
    return timestamp.replace(minute=0, second=0, microsecond=0)


def _parse_hour(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("invalid hour value")
    if isinstance(value, (int, float)):
        hour = int(value)
    elif isinstance(value, str):
        match = re.search(r"\d{1,2}", value)
        if not match:
            raise ValueError("invalid hour value")
        hour = int(match.group())
    else:
        raise TypeError("unsupported hour value")
    if not 0 <= hour <= 24:
        raise ValueError("hour must be between 0 and 24")
    return hour


def _first_number(
    record: dict[str, Any],
    keys: tuple[str, ...],
    required: bool,
) -> float | None:
    for key in keys:
        if key in record and record[key] is not None:
            return _parse_number(record[key])
    if required:
        raise ValueError(f"missing required price field ({', '.join(keys)})")
    return None


def _parse_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid price")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        normalized = value.strip().replace(" ", "")
        if "," in normalized and "." in normalized:
            if normalized.rfind(",") > normalized.rfind("."):
                normalized = normalized.replace(".", "").replace(",", ".")
            else:
                normalized = normalized.replace(",", "")
        elif "," in normalized:
            normalized = normalized.replace(",", ".")
        result = float(normalized)
    else:
        raise TypeError("unsupported price value")
    if not math.isfinite(result):
        raise ValueError("price must be finite")
    return result
