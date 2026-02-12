from typing import List

from base_dao import user_dao
from fastapi import APIRouter, HTTPException
from user_model import User, UserCreate, UserUpdate

router = APIRouter(prefix="/users", tags=["users managerment"])


@router.post("/", response_model=User)
def create_user(user: UserCreate):
    existing_user = user_dao.get_by_field("username", user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="user is already exist")

    db_user = User(**user.model_dump())
    return user_dao.create(db_user)


@router.get("/", response_model=List[User])
def get_all_users():
    return user_dao.get_all()


@router.get("/{user_id}", response_model=User)
def get_user(user_id: int):
    user = user_dao.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user is not exist")
    return user


@router.put("/{user_id}", response_model=User)
def update_user(user_id: int, user_update: UserUpdate):
    user = user_dao.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user is not exist")

    # exclude_unset=True: 只更新提供的字段
    update_data = user_update.model_dump(exclude_unset=True)
    return user_dao.update(user_id, update_data)


@router.delete("/{user_id}")
def delete_user(user_id: int):
    ok = user_dao.delete(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="user is not exist")
    return {"message": "delete user success"}
