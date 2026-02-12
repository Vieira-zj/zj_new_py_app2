from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    username: str = "postgres"
    password: str
    database: str

    class Config:
        env_prefix = "DB_"


# 获取配置
db_config = DatabaseSettings()  # type: ignore
