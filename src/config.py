import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Environment
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

# API Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "your-dev-api-key")  # Used for API authentication

# Storage Configuration
STORAGE_TYPE = os.environ.get("STORAGE_TYPE", "local")  # 'local' or 's3'
S3_BUCKET = os.environ.get("S3_BUCKET", "pump-anomaly-models")
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
METRICS_HISTORY_PATH = os.path.join(DATA_DIR, "metrics_history.csv")

# Logging Configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")

# Security
SECRET_KEY = os.environ.get("SECRET_KEY", "your-dev-secret-key")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Create necessary directories
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
