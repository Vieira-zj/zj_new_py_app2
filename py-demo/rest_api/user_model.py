from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel

# DB Model


class User(SQLModel, table=True):
    __tablename__ = "users"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(max_length=50, unique=True)
    email: str = Field(max_length=100, unique=True)
    password: str = Field(max_length=100)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        # 序列化时排除 password
        exclude = {"password"}


# Request Model


class UserCreate(SQLModel):
    username: str = Field(max_length=50)
    email: str = Field(max_length=100)
    password: str = Field(min_length=6)


class UserUpdate(SQLModel):
    username: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None
