import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.core.db.postgress_db import engine, Base, SessionLocal
from app.api.router import router as auth_router  # Add your actual routers here

class MainApp:
    def __init__(self):
        load_dotenv(override=True)
        if (
            os.getenv("isDevelopmentMode") == "enabled"
            and os.getenv("ENV") != "development"
        ):
            logging.error("Development mode is enabled  but ENV is not development.")
            exit(1)

        self.app = FastAPI(title="RepoPlay API", version="1.0")
        self.setup_cors()
        self.init_database()
        self.include_routers()
        self.add_health_check()

    def setup_cors(self):
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
