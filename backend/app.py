"""Stateless FastAPI gateway and examples for OpenAI-compatible providers."""
import logging
import os
import pdb
import re
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from starlette.exceptions import HTTPException as StarletteHTTPException

from mock_database import mock_db_api
from routes import api
from settings import configure_app, positive
from structured_outputs import structured_api

logger = logging.getLogger("llm_api")


class RequestBodyLimitMiddleware:
    def __init__(self, app, config_values: dict):
        self.app = app
        self.config_values = config_values

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_bytes = self.config_values["MAX_REQUEST_BYTES"]
        content_length = dict(scope.get("headers", [])).get(b"content-length", b"")
        if content_length.isdigit() and int(content_length) > max_bytes:
            await self._reject(send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                await self.app(scope, receive, send)
                return
            body.extend(message.get("body", b""))
            if len(body) > max_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break

        body_sent = False

        async def replay_body():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay_body, send)

    async def _reject(self, send):
        body = b'{"error":"Request body is too large"}'
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})


def create_app(config=None, openai_client=None):
    application = FastAPI(
        title="Parley LLM API",
        version="1.1.0",
        description="Chat completions, structured outputs, and function tool-calling examples.",
        lifespan=_lifespan,
    )
    configure_app(application, config)
    application.state.mock_database = {}
    application.state.pending_approvals = {}
    application.state.provider_clients = {}
    if openai_client is not None:
        application.state.provider_clients = {
            provider: openai_client
            for provider in application.state.config_values["PROVIDER_SETTINGS"]
        }

    application.add_middleware(
        RequestBodyLimitMiddleware,
        config_values=application.state.config_values,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=application.state.config_values["ALLOWED_ORIGINS"],
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID", "X-Demo-User-ID"],
        expose_headers=["Retry-After", "X-Request-ID"],
        allow_credentials=False,
    )

    @application.middleware("http")
    async def request_policy_and_metadata(request: Request, call_next):
        config_values = request.app.state.config_values
        origin = request.headers.get("Origin")
        if origin and origin not in config_values["ALLOWED_ORIGINS"]:
            return JSONResponse({"error": "This browser origin is not allowed"}, status_code=403)

        request_id = request.headers.get("X-Request-ID", "")
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id):
            request_id = f"req-{time.time_ns()}"
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_failed request_id=%s method=%s path=%s",
                request_id, request.method, request.url.path,
            )
            raise
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_complete request_id=%s method=%s path=%s status_code=%s latency_ms=%.2f",
            request_id, request.method, request.url.path, response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            {"error": "Invalid request", "request_id": request.state.request_id},
            status_code=400,
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException):
        message = {
            404: "Endpoint not found",
            405: "Method not allowed",
        }.get(exc.status_code, "Request failed")
        return JSONResponse({"error": message}, status_code=exc.status_code)

    application.include_router(api)
    application.include_router(mock_db_api)
    application.include_router(structured_api)
    return application


@asynccontextmanager
async def _lifespan(application: FastAPI):
    provider_clients = application.state.provider_clients
    owns_clients = not provider_clients
    try:
        if owns_clients:
            config = application.state.config_values
            connect_timeout, read_timeout = config["UPSTREAM_TIMEOUT"]
            for provider, provider_config in config["PROVIDER_SETTINGS"].items():
                if not provider_config["api_key"]:
                    key_name = "PROVIDER_API_KEY" if provider == "DEFAULT" else f"{provider}_API_KEY"
                    raise ValueError(f"Set {key_name} in the backend environment")
                provider_clients[provider] = AsyncOpenAI(
                    api_key=provider_config["api_key"],
                    base_url=provider_config["base_url"],
                    timeout=httpx.Timeout(timeout=read_timeout, connect=connect_timeout),
                    max_retries=1,
                )
        yield
    finally:
        if owns_clients:
            for client in provider_clients.values():
                await client.close()


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = positive(os.getenv("PORT", "8000"), "PORT", int)
    if port > 65535:
        raise ValueError("PORT must be between 1 and 65535")
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=port,
        reload=True,
    )
