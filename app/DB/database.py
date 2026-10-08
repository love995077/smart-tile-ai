import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# 1. The Connection String
# Format: mysql+pymysql://username:password@host:port/database_name
# Production sets DATABASE_URL (see .env / docker-compose.yml). The fallback is the
# local phpMyAdmin default: user 'root' with an empty password.
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:@127.0.0.1:3306/bharat_ceramic")

# 2. Create the Database Engine
# pool_pre_ping drops dead pooled connections (e.g. after MySQL restarts);
# connect_timeout keeps an unreachable server from hanging requests.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 3},
)

# 3. Create a Local Sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Create the Base Model class
Base = declarative_base()

# 5. Helper function to use in your routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()