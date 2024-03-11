from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import router
from database import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# todo
# input -> url -> get steamid64 -> Error check -> get library and play time -> Error checklr ->
# database 연결
# model load
# model
# latency checkㅂ