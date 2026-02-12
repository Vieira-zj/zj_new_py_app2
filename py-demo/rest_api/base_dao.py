from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from settings import db_config
from sqlalchemy import create_engine, select
from sqlmodel import Session, SQLModel
from user_model import User

# 泛型类型
ModelType = TypeVar("ModelType", bound=SQLModel)


class BaseDao(Generic[ModelType]):

    def __init__(self, model: Type[ModelType]):
        self.model = model
        database_url = f"postgresql://{db_config.username}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
        self.engine = create_engine(database_url)

    def create(self, obj_in: ModelType) -> ModelType:
        with Session(self.engine) as session:
            db_obj = self.model.model_validate(obj_in)
            session.add(db_obj)
            session.commit()
            session.refresh(db_obj)
            return db_obj

    def get_by_id(self, user_id: int) -> Optional[ModelType]:
        with Session(self.engine) as session:
            statement = select(self.model).where(self.model.id == user_id)  # type: ignore
            result = session.exec(statement).first()  # type: ignore
            return result

    def get_all(self) -> List[ModelType]:
        with Session(self.engine) as session:
            statement = select(self.model)
            result = session.exec(statement).all()  # type: ignore
            return result

    def update(self, user_id: int, obj_in: Dict[str, Any]) -> Optional[ModelType]:
        with Session(self.engine) as session:
            statement = select(self.model).where(self.model.id == user_id)  # type: ignore
            db_obj = session.exec(statement).first()  # type: ignore

            if db_obj:
                for key, value in obj_in.items():
                    if hasattr(db_obj, key):
                        setattr(db_obj, key, value)
                session.add(db_obj)
                session.commit()
                session.refresh(db_obj)
            return db_obj

    def delete(self, user_id: int) -> bool:
        with Session(self.engine) as session:
            statement = select(self.model).where(self.model.row_id == user_id)  # type: ignore
            db_obj = session.exec(statement).first()  # type: ignore
            if not db_obj:
                return False

            session.delete(db_obj)
            session.commit()
            return True

    def get_by_field(self, field_name: str, value: Any) -> Optional[ModelType]:
        with Session(self.engine) as session:
            statement = select(self.model).where(
                getattr(self.model, field_name) == value
            )
            result = session.exec(statement).first()  # type: ignore
            return result

    def get_page(self, skip: int = 0, limit: int = 20) -> List[ModelType]:
        with Session(self.engine) as session:
            statement = select(self.model).offset(skip).limit(limit)
            result = session.exec(statement).all()  # type: ignore
            return result

    def get_by_conditions(self, conditions: Dict[str, Any]) -> List[ModelType]:
        with Session(self.engine) as session:
            statement = select(self.model)
            for field, value in conditions.items():
                statement = statement.where(getattr(self.model, field) == value)
            result = session.exec(statement).all()  # type: ignore
            return result


user_dao = BaseDao[User](User)
