# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import router as api_router
from .config import settings

app = FastAPI(title="Rat Surveillance Backend", version="0.1.0")

# CORS: allow local frontend later (React/Vue/etc)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.mount("/artifacts", StaticFiles(directory=str(settings.PROCESSED_DIR)), name="artifacts")
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOADS_DIR)), name="uploads")
app.mount("/", StaticFiles(directory=str(settings.BASE_DIR.parent / "frontend"), html=True), name="frontend")


@app.get("/")
def root():
    return {"message": "Rat Surveillance Backend is running"}
