from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# 1. The Connection String
# Format: mysql+pymysql://username:password@host:port/database_name
# Default phpMyAdmin uses user 'root' and an empty password
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:@127.0.0.1:3306/bharat_ceramic"

# 2. Create the Database Engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

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