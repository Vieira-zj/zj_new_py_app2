from fastapi import APIRouter

test_router = APIRouter(prefix="/test", tags=["fast api test"])


# curl "http://localhost:8081/test/sleep?duration=0.3"
@test_router.get("/sleep")
async def sleep_handler(duration: float = 0.1):
    import asyncio

    await asyncio.sleep(duration)
    return {"message": f"slept for {duration:.2f} seconds"}
