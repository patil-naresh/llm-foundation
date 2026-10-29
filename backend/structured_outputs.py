import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from provider_errors import provider_error_response
from providers import client_for_model
from schemas import Person, StructuredOutputRequest

structured_api = APIRouter(prefix="/v1/structured-outputs", tags=["structured outputs"])
logger = logging.getLogger("llm_api")

PERSON_PROMPT = "Give me the name, age and city of a fictional person."


@structured_api.post("/plain")
async def unstructured_output(payload: StructuredOutputRequest, request: Request):
    return await _create_text_completion(payload, request, json_prompt=False)


@structured_api.post("/json-prompt")
async def json_prompt_output(payload: StructuredOutputRequest, request: Request):
    return await _create_text_completion(payload, request, json_prompt=True)


@structured_api.post("/person")
async def person_output(payload: StructuredOutputRequest, request: Request):
    model = payload.model or request.app.state.config_values["DEFAULT_MODEL"]
    if model not in request.app.state.config_values["MODEL_ALLOWLIST"]:
        return JSONResponse({"error": "Unknown model"}, status_code=400)
    config = request.app.state.config_values
    try:
        result = await client_for_model(request, model).chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": payload.system_instructions or config["SYSTEM_INSTRUCTIONS"],
                },
                {"role": "user", "content": payload.query or PERSON_PROMPT},
            ],
            response_format=Person,
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIStatusError) as exc:
        return provider_error_response(request, model, exc)
    except Exception as exc:
        logger.error(
            "structured_output_failed request_id=%s error_type=%s",
            request.state.request_id, type(exc).__name__,
        )
        return JSONResponse(
            {
                "error": "Could not process structured model output.",
                "request_id": request.state.request_id,
            },
            status_code=502,
        )

    if not result.choices or result.choices[0].message.parsed is None:
        return JSONResponse(
            {
                "error": "The provider did not return a valid Person object.",
                "request_id": request.state.request_id,
            },
            status_code=502,
        )
    return {
        "request_id": request.state.request_id,
        "model": result.model or model,
        "person": result.choices[0].message.parsed.model_dump(),
    }


async def _create_text_completion(
    payload: StructuredOutputRequest,
    request: Request,
    *,
    json_prompt: bool,
):
    model = payload.model or request.app.state.config_values["DEFAULT_MODEL"]
    config = request.app.state.config_values
    if model not in config["MODEL_ALLOWLIST"]:
        return JSONResponse({"error": "Unknown model"}, status_code=400)

    system_prompt = payload.system_instructions or config["SYSTEM_INSTRUCTIONS"]
    if json_prompt:
        system_prompt = f"{system_prompt} Respond in JSON format."
    try:
        result = await client_for_model(request, model).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload.query},
            ],
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIStatusError) as exc:
        return provider_error_response(request, model, exc)
    except Exception as exc:
        logger.error(
            "structured_output_failed request_id=%s error_type=%s",
            request.state.request_id, type(exc).__name__,
        )
        return JSONResponse(
            {"error": "Could not process model output.", "request_id": request.state.request_id},
            status_code=500,
        )

    answer = result.choices[0].message.content if result.choices else None
    if answer is None:
        return JSONResponse(
            {
                "error": "The provider returned no text response.",
                "request_id": request.state.request_id,
            },
            status_code=502,
        )
    return {
        "request_id": request.state.request_id,
        "model": result.model or model,
        "mode": "json-prompt" if json_prompt else "plain",
        "answer": answer,
    }
