from contextlib import asynccontextmanager
from fastapi import FastAPI
from .routers import users, auth, events, upload, profile
from .database import initialize_db, client

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await initialize_db()
        yield
    finally:
        client.close()

app = FastAPI(lifespan=lifespan)

app.include_router(users.router)
app.include_router(auth.router)
app.include_router(events.router)
app.include_router(upload.router)
app.include_router(profile.router)

@app.get("/")
def root():
    return {"message": "Root connected successfully"}
