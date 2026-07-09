from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "EPİAŞ PTF Forecasting MVP"
    app_version: str = "0.1.0"
    environment: str = "development"
    database_url: str = "postgresql+psycopg://pepias:pepias@db:5432/pepias"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    epias_base_url: str = "https://seffaflik.epias.com.tr"
    epias_auth_url: str = "https://giris.epias.com.tr"
    epias_username: str | None = None
    epias_password: str | None = None
    epias_request_timeout: float = Field(default=30.0, gt=0)
    epias_max_retries: int = Field(default=3, ge=0, le=10)
    epias_ptf_endpoint: str = "/electricity-service/v1/markets/dam/data/mcp"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def epias_credentials_configured(self) -> bool:
        return bool(self.epias_username and self.epias_password)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
