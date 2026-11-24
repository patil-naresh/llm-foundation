import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_BASE_URL = os.getenv(
    "BACKEND_BASE_URL",
    "http://127.0.0.1:5000"
)

BACKEND_HEALTH_PATH = os.getenv(
    "BACKEND_HEALTH_PATH",
    "/health"
)

BACKEND_MODELS_PATH = os.getenv(
    "BACKEND_MODELS_PATH",
    "/v1/models"
)

BACKEND_CHAT_PATH = os.getenv(
    "BACKEND_CHAT_PATH",
    "/v1/chat/completions"
)

BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")