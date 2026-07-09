from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class EpiasHealthResponse(BaseModel):
    epias_base_url: str
    credentials_configured: bool
    client_ready: bool


class EpiasTestPostRequest(BaseModel):
    endpoint: str
    payload: dict[str, Any] = Field(default_factory=dict)
    use_auth: bool = True

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        if not value.startswith("/") or "://" in value:
            raise ValueError("endpoint must be a relative path starting with '/'")
        return value


class EpiasTestPostResponse(BaseModel):
    endpoint: str
    status_code: int
    raw_response_id: int
    result: Any


class PtfIngestionRequest(BaseModel):
    start_date: date
    end_date: date
    chunk_days: int = Field(default=30, ge=1, le=366)

    @model_validator(mode="after")
    def validate_date_range(self) -> "PtfIngestionRequest":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class PtfIngestionSummary(BaseModel):
    start_date: str
    end_date: str
    chunks_processed: int
    records_fetched: int
    records_inserted_or_updated: int
    missing_hours: list[str]
    errors: list[str]


class PtfStatusResponse(BaseModel):
    total_rows: int
    min_timestamp: datetime | None
    max_timestamp: datetime | None
    latest_updated_at: datetime | None
