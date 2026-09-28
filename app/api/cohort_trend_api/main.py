
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:  # load .env (DB credentials, API_PREFIX, ALLOWED_ORIGINS); searches upward from here
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.api.cohort_trend_api.routers import router as trends_cohorts_router  # noqa: E402

app = FastAPI(
    title=os.getenv("APP_NAME", "Customer Intelligence Platform"),
    version="1.0.0",
)

# Allow the frontend (e.g. http://localhost:3000) to call the API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(trends_cohorts_router, prefix=os.getenv("API_PREFIX", "/api/v1"))


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
