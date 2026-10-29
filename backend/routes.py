import json
import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from provider_errors import provider_error_response
from providers import client_for_model
from schemas import ApprovalDecisionRequest, ChatCompletionRequest, ChatRequest, ToolCallRequest
from tool_calls import (
    InvalidToolCall,
    MissingDemoUserIdentity,
    TOOLS,
    map_tool_calls,
    normalize_demo_user_id,
)
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

api = APIRouter()
logger = logging.getLogger("llm_api")


@api.get("/health")
async def health():
    return {"status": "ok"}


@api.get("/v1/models")
async def list_models(request: Request):
    models = request.app.state.config_values["MODEL_ALLOWLIST"]
    return {"object": "list", "data": [{"id": model, "object": "model"} for model in models]}


@api.post("/v1/approvals/{approval_id}")
async def decide_approval(
    approval_id: str,
    payload: ApprovalDecisionRequest,
    request: Request,
):
    user_id = normalize_demo_user_id(request.headers.get("X-Demo-User-ID"))
    if user_id is None:
        return JSONResponse(
            {
                "error": "A valid demo user ID is required to decide this approval.",
                "request_id": request.state.request_id,
            },
            status_code=401,
        )

    approval = request.app.state.pending_approvals.get(approval_id)
    if not isinstance(approval, dict) or approval.get("owner_user_id") != user_id:
        return JSONResponse(
            {"error": "Approval request not found.", "request_id": request.state.request_id},
            status_code=404,
        )
    if approval.get("approval_status") != "pending":
        return JSONResponse(
            {"error": "Approval request has already been decided.", "request_id": request.state.request_id},
            status_code=409,
        )

    if not payload.approved:
        approval["approval_status"] = "rejected"
        return {"approval_id": approval_id, "approval_status": "rejected"}

    refund_id = str(uuid4())
    refund = {
        "refund_id": refund_id,
        "order_id": approval["order_id"],
        "amount_cents": approval["amount_cents"],
        "reason": approval["reason"],
        "status": "issued",
        "mock": True,
    }
    request.app.state.mock_database[f"refunds/{refund_id}"] = refund
    approval["approval_status"] = "approved"
    approval["result"] = refund
    return {
        "approval_id": approval_id,
        "approval_status": "approved",
        "result": refund,
    }


@api.post("/v1/chat")
async def basic_chat(payload: ChatRequest, request: Request):
    config = request.app.state.config_values
    query = payload.query.strip()
    request_id = request.state.request_id
    if not query:
        return JSONResponse(
            {"error": "query must contain non-whitespace characters.", "request_id": request_id},
            status_code=422,
        )
    if len(query) > config["MAX_QUERY_LENGTH"]:
        return JSONResponse(
            {
                "error": (
                    "query exceeds the configured limit of "
                    f"{config['MAX_QUERY_LENGTH']} characters."
                ),
                "request_id": request_id,
            },
            status_code=413,
        )

    model = config["DEFAULT_MODEL"]
    started = time.perf_counter()
    try:
        result = await client_for_model(request, model).responses.create(
            model=model,
            instructions=payload.system_instructions or config["SYSTEM_INSTRUCTIONS"],
            input=query,
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIStatusError) as exc:
        return provider_error_response(request, model, exc)
    except Exception as exc:
        logger.error(
            "basic_chat_failed request_id=%s error_type=%s",
            request_id, type(exc).__name__,
        )
        return JSONResponse({"error": "Internal server error", "request_id": request_id}, status_code=500)

    usage = _responses_usage(getattr(result, "usage", None))
    return {
        "request_id": request_id,
        "answer": result.output_text,
        "model": result.model or model,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "usage": usage,
    }

@api.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    config = request.app.state.config_values
    model = payload.model or config["DEFAULT_MODEL"]
    if model not in config["MODEL_ALLOWLIST"]:
        return JSONResponse({"error": "Unknown model"}, status_code=400)
    options: dict[str, Any] = {
        "model": model,
        "messages": [message.model_dump() for message in payload.messages],
        "stream": False,
        "tools": TOOLS,
    }
    if payload.temperature is not None:
        options["temperature"] = payload.temperature
    if payload.max_tokens is not None:
        options["max_tokens"] = payload.max_tokens
    options.update(payload.model_extra or {})

    try:
        client = client_for_model(request, model)
        result = await client.chat.completions.create(**options)
        messages = options["messages"]
        tool_results: list[dict[str, Any]] = []
        for _ in range(5):
            if not result.choices:
                return _chat_provider_shape_error(request)
            assistant_message = result.choices[0].message
            tool_calls = assistant_message.tool_calls or []
            if not tool_calls:
                break
            if len(tool_calls) > 10:
                return _chat_provider_shape_error(request)
            messages.append(assistant_message.model_dump(exclude_none=True))
            try:
                tool_messages, executions = map_tool_calls(
                    tool_calls,
                    request.app.state.mock_database,
                    request.headers.get("X-Demo-User-ID"),
                    request.app.state.pending_approvals,
                )
            except MissingDemoUserIdentity:
                return JSONResponse(
                    {
                        "error": "A valid demo user ID is required for this tool.",
                        "request_id": request.state.request_id,
                    },
                    status_code=401,
                )
            except InvalidToolCall:
                return _chat_provider_shape_error(request)
            messages.extend(tool_messages)
            tool_results.extend(executions)
            result = await client.chat.completions.create(**{**options, "messages": messages})
        else:
            if result.choices and result.choices[0].message.tool_calls:
                return _chat_provider_shape_error(request)
    except (APITimeoutError, RateLimitError, APIConnectionError, APIStatusError) as exc:
        return provider_error_response(request, model, exc)
    except Exception as exc:
        logger.error(
            "chat_completion_failed request_id=%s model=%s error_type=%s",
            request.state.request_id, model, type(exc).__name__,
        )
        return JSONResponse(
            {"error": "Internal server error", "request_id": request.state.request_id},
            status_code=500,
        )

    if payload.stream:
        return StreamingResponse(
            _stream_final_completion(result, request.state.request_id, tool_results),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )
    response = result.model_dump(mode="json", exclude_none=True)
    if tool_results:
        response["tool_results"] = tool_results
    return JSONResponse(content=response)


def _chat_provider_shape_error(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "error": "The provider returned an invalid tool-calling response.",
            "request_id": request.state.request_id,
        },
        status_code=502,
    )


def _responses_usage(usage_obj: Any) -> dict[str, int] | None:
    if usage_obj is None:
        return None
    usage: dict[str, int] = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage_obj, field, None)
        if value is not None:
            usage[field] = value
    return usage or None


async def _stream_final_completion(
    result: Any,
    request_id: str,
    tool_results: list[dict[str, Any]],
):
    if not result.choices:
        yield (
            "data: "
            + json.dumps(
                {"error": "The provider returned an empty completion.", "request_id": request_id}
            )
            + "\n\n"
        )
        yield "data: [DONE]\n\n"
        return

    choice = result.choices[0]
    message = choice.message
    data = {
        "id": getattr(result, "id", "chatcmpl-final"),
        "object": "chat.completion.chunk",
        "created": getattr(result, "created", int(time.time())),
        "model": getattr(result, "model", None),
        "tool_results": tool_results,
        "choices": [{
            "index": getattr(choice, "index", 0),
            "delta": {
                key: value
                for key, value in {
                    "role": getattr(message, "role", "assistant"),
                    "content": getattr(message, "content", None),
                }.items()
                if value is not None
            },
            "finish_reason": getattr(choice, "finish_reason", None) or "stop",
        }],
    }
    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
