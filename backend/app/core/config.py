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

    # Optional: free instant-signup key at fredaccount.stlouisfed.org/apikeys -
    # unlocks real macro data (yield curve, unemployment, inflation) in Contexto.
    # Left unset, that section is simply omitted rather than erroring.
    fred_api_key: str | None = None

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
