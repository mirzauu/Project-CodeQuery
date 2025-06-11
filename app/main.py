import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.db.postgress_db import engine, Base, SessionLocal
from app.api.router import router as auth_router  # Add your actual routers here

# # Setup logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
# )
# logger = logging.getLogger(__name__)


class PreflightCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Handle only OPTIONS requests (preflight)
        if request.method == "OPTIONS":
            response = Response(status_code=200)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Max-Age"] = "600"  # 10 minutes
            return response

        # Let other requests pass through
        response = await call_next(request)
        return response

class MainApp:
    def __init__(self):
        load_dotenv(override=True)


        self.app = FastAPI(title="RepoPlay API", version="1.0")
        self.setup_cors()
        self.init_database()
        self.include_routers()
        self.add_health_check()

    def setup_cors(self):
        self.app.add_middleware(PreflightCacheMiddleware)
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Update this for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def init_database(self):
        # If you prefer alembic only, comment this out
        Base.metadata.create_all(bind=engine)

    def include_routers(self):
        self.app.include_router(auth_router, prefix="/api/v1", tags=["Users"])

    def add_health_check(self):
        @self.app.get("/health", tags=["Health"])
        def health():
            return {"status": "ok"}

    def run(self):
        return self.app


main_app = MainApp()
app = main_app.run()
