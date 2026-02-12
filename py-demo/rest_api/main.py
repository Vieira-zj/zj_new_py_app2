from fastapi import FastAPI
from settings import db_config
from sqlmodel import SQLModel, create_engine
from user_api import router as user_router

database_url = f"postgresql://{db_config.username}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
engine = create_engine(database_url)


def create_db_and_tables():
    SQLModel.metadata.create_all(
        engine,
    )


app = FastAPI(title="users managerment demo")
app.include_router(user_router)


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)


if __name__ == "__main__":
    main()
