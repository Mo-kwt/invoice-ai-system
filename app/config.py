from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4.1-mini"
    database_url: str = "sqlite:///./invoice_ai.db"
    upload_dir: str = "storage/uploads"
    processed_dir: str = "storage/processed"
    export_dir: str = "storage/exports"
    max_file_size_mb: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()