from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://quantumalpha:quantumalpha@localhost:5432/quantumalpha"

    cors_origins: str = "http://localhost:5173"

    market_data_provider: str = "yfinance"
    risk_free_rate: float = 0.04

    # Reconstruction (2026-09), Fase 7: the read-only Gemini narrative layer
    # over the gate (see GeminiNarrator's own docstring) - `None` (the
    # default, and what .env.example ships) keeps it fully inert, no network
    # call ever attempted and no data ever sent to Google. Setting this is a
    # deliberate, owner-made decision (cost, privacy), same as activating the
    # Fase 2 cron jobs on Render - never something to set from code.
    gemini_api_key: str | None = None

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
