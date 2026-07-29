from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from Weft.config.settings import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass
