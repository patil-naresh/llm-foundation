import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

logger = logging.getLogger("llm_api")


def provider_error_response(request: Request, model: str, exc: Exception) -> JSONResponse:
    request_id = request.state.request_id
    if isinstance(exc, APITimeoutError):
        status_code, message = 504, "AI provider timed out, try again shortly"
        logger.warning("provider_timeout request_id=%s model=%s", request_id, model)
    elif isinstance(exc, RateLimitError):
        status_code, message = 429, "Too many requests, try again shortly"
        logger.warning("provider_rate_limited request_id=%s model=%s", request_id, model)
    elif isinstance(exc, APIConnectionError):
        status_code, message = 502, "AI provider unavailable"
        logger.warning("provider_connection_error request_id=%s model=%s", request_id, model)
    elif isinstance(exc, APIStatusError):
        status_code, message = 502, "AI provider unavailable"
        logger.error(
            "provider_error request_id=%s model=%s status=%s",
            request_id, model, exc.status_code,
        )
    else:
        status_code, message = 500, "Internal server error"
        logger.error(
            "provider_unexpected_error request_id=%s model=%s error_type=%s",
            request_id, model, type(exc).__name__,
        )

    response = JSONResponse(
        {"error": message, "request_id": request_id},
        status_code=status_code,
    )
    if isinstance(exc, RateLimitError):
        retry_after = getattr(exc, "response", None)
        retry_after = retry_after.headers.get("Retry-After", "") if retry_after else ""
        if retry_after.isdigit() and len(retry_after) <= 8:
            response.headers["Retry-After"] = retry_after
    return response
