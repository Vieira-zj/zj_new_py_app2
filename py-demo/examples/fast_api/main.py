import time
from typing import Awaitable, Callable

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from router import test_router
from starlette.status import HTTP_404_NOT_FOUND

router = APIRouter(tags=["fast api demo"])


# curl "http://localhost:8081/healthz"
@router.get("/healthz")
async def healthz_handler():
    return {"message": "ok"}


app = FastAPI(title="users managerment demo")
app.include_router(router)
app.include_router(test_router)


@app.middleware("http")
async def add_process_time_header(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
):
    start = time.perf_counter()
    response = await call_next(request)

    process_time = f"{(time.perf_counter() - start):.4f}"
    response.headers["X-Process-Time"] = process_time
    print(f"request to {request.url.path} processed in {process_time} seconds")
    return response


# curl "http://localhost:8081/"
@app.get("/")
async def home_handler():
    return {"message": "welcome to fast api"}


# curl "http://localhost:8081/not_found"
@app.exception_handler(HTTP_404_NOT_FOUND)
async def not_found_hanlder(request: Request, err: HTTPException):
    _ = err
    return JSONResponse(
        status_code=HTTP_404_NOT_FOUND,
        content={
            "message": "the resource you are looking for is not found",
            "path": request.url.path,
        },
    )


def main():
    import uvicorn

    port = 8081
    print(f"start fast api server at: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
