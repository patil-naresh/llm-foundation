import httpx

from config import (
    BACKEND_BASE_URL,
    BACKEND_HEALTH_PATH,
    BACKEND_MODELS_PATH,
    BACKEND_CHAT_PATH,
    BACKEND_API_KEY,
)


class BackendClient:

    def __init__(self):
        self.base_url = BACKEND_BASE_URL.rstrip("/")
        self.headers = {}

        if BACKEND_API_KEY:
            self.headers["Authorization"] = (
                f"Bearer {BACKEND_API_KEY}"
            )

    async def get(self, path: str):
        url = f"{self.base_url}{path}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                headers=self.headers
            )

            response.raise_for_status()

            return response.json()

    async def post(self, path: str, payload: dict):

        url = f"{self.base_url}{path}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers=self.headers
            )

            response.raise_for_status()

            return response.json()

    async def check_health(self):
        return await self.get(BACKEND_HEALTH_PATH)

    async def get_models(self):

        return await self.get(BACKEND_MODELS_PATH)

    async def chat(self, model: str, message: str):

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        }

        return await self.post(
            BACKEND_CHAT_PATH,
            payload
        )