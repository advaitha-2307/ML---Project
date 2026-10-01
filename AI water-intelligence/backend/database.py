import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("water_intelligence.database")

# Environment-configurable database URL (PostgreSQL default)
DEFAULT_PG_URL = "postgresql://postgres:postgres@localhost:5432/water_intelligence"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_PG_URL)

Base = declarative_base()

active_engine = None
SessionLocal = None
is_sqlite_fallback = False

def create_db_engine():
    global active_engine, SessionLocal, is_sqlite_fallback
    
    # First, attempt connecting with the configured DATABASE_URL (PostgreSQL)
    try:
        if DATABASE_URL.startswith("postgresql"):
            test_engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                connect_args={"connect_timeout": 3}
            )
            # Try a quick test connection
            with test_engine.connect() as conn:
                pass
            logger.info("Connected successfully to PostgreSQL database: %s", DATABASE_URL.split("@")[-1])
            active_engine = test_engine
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=active_engine)
            is_sqlite_fallback = False
            return active_engine
    except Exception as e:
        logger.warning(
            "Could not connect to PostgreSQL at '%s': %s.\n"
            "Falling back to local SQLite database (predictions.db) to ensure the platform remains fully functional.\n"
            "To connect to your PostgreSQL instance, set DATABASE_URL in .env (e.g. postgresql://user:password@localhost:5432/water_intelligence).",
            DATABASE_URL, e
        )

    # Fallback to local SQLite
    sqlite_url = "sqlite:///./predictions.db"
    active_engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=active_engine)
    is_sqlite_fallback = True
    logger.info("Using SQLite database at %s", sqlite_url)
    return active_engine

# Initialize engine and session
engine = create_db_engine()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    import backend.models  # noqa: F401
    Base.metadata.create_all(bind=active_engine)
    logger.info("Database schema initialized successfully.")

# Pre-initialize schema on import
try:
    init_db()
except Exception as e:
    logger.warning("Auto schema init deferred: %s", e)

