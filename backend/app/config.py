from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL
from sqlalchemy.engine import make_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str = "recipetriage"
    postgres_password: SecretStr | None = None
    database_dsn: SecretStr | None = Field(default=None, validation_alias='DATABASE_URL')
    postgres_db: str = "recipetriage"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    @model_validator(mode='after')
    def database_configured(self):
        if self.database_dsn is None and self.postgres_password is None:
            raise ValueError('Configure DATABASE_URL or POSTGRES_PASSWORD.')
        if self.database_dsn is not None:
            try:
                parsed = make_url(self.database_dsn.get_secret_value())
                if parsed.drivername not in {'postgres', 'postgresql', 'postgresql+psycopg'} or not parsed.host or not parsed.database:
                    raise ValueError()
            except Exception:
                raise ValueError('DATABASE_URL must be a PostgreSQL connection URL with a host and database.') from None
        return self

    @property
    def database_url(self) -> URL:
        if self.database_dsn is not None:
            return make_url(self.database_dsn.get_secret_value()).set(drivername='postgresql+psycopg')
        # URL.create safely handles punctuation in credentials.
        return URL.create(
            "postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )
