import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access environment variables
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
ENV = os.getenv("ENV")
DEBUG = os.getenv("DEBUG") == 'True'
GMAIL_USERNAME = os.getenv("GMAIL_USERNAME")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
BEST_MODEL_PATH=os.getenv("BEST_MODEL_PATH")
SHAPE_PREDICTOR_PATH=os.getenv("SHAPE_PREDICTOR_PATH")