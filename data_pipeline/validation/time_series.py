from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")


def find_missing_hourly_timestamps(
    start_datetime: datetime,
    end_datetime: datetime,
    existing_timestamps: Iterable[datetime],
) -> list[datetime]:
    start = _ensure_aware(start_datetime).replace(minute=0, second=0, microsecond=0)
    end = _ensure_aware(end_datetime).replace(minute=0, second=0, microsecond=0)
    if end < start:
        raise ValueError("end_datetime must be on or after start_datetime")

    existing_utc = {
        _ensure_aware(timestamp)
        .astimezone(timezone.utc)
        .replace(minute=0, second=0, microsecond=0)
        for timestamp in existing_timestamps
    }
    output_timezone = start.tzinfo or ISTANBUL_TIMEZONE
    current_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    missing: list[datetime] = []

    while current_utc <= end_utc:
        if current_utc not in existing_utc:
            missing.append(current_utc.astimezone(output_timezone))
        current_utc += timedelta(hours=1)

    return missing


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=ISTANBUL_TIMEZONE)
    return value

