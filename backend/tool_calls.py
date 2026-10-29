import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, MutableMapping
from uuid import UUID, uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from schemas import (
    OrderStatusArguments,
    RefundApprovalArguments,
    SupportTicketDetailsArguments,
    SupportTicketArguments,
    WeatherArguments,
)

logger = logging.getLogger("llm_api")

MOCK_ORDERS = {
    "A123": {
        "order_id": "A123",
        "status": "Shipped",
        "estimated_delivery": "2026-10-29",
    },
    "B456": {
        "order_id": "B456",
        "status": "Processing",
        "estimated_delivery": None,
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current mock weather for a city in India.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city to get weather for"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the status of a mock order by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The ID of the order to check"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a support ticket for a given order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The ID of the order for which to create a ticket"},
                    "issue_description": {"type": "string", "description": "A description of the issue"}
                },
                "required": ["order_id", "issue_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_support_ticket_details",
            "description": "Get the details of a support ticket by its ID.",    
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "The ID of the support ticket to retrieve"}
                },
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_refund",
            "description": (
                "Request a mock refund. This only creates a pending approval request; "
                "never claim the refund was issued until a human approves it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order to refund"},
                    "amount_cents": {
                        "type": "integer",
                        "description": "Refund amount in cents",
                    },
                    "reason": {"type": "string", "description": "Why the refund is requested"},
                },
                "required": ["order_id", "amount_cents", "reason"],
            },
        },
    },
]

def get_support_ticket_details(
    ticket_id: str,
    database: MutableMapping[str, Any],
    user_id: str,
) -> dict[str, Any]:
    ticket = database.get(f"support_tickets/{ticket_id}")
    if not isinstance(ticket, dict) or ticket.get("owner_user_id") != user_id:
        return {
            "ticket_id": ticket_id,
            "message": "No support ticket found with the provided ticket ID.",
            "found": False,
        }
    return {
        "found": True,
        **{key: value for key, value in ticket.items() if key != "owner_user_id"},
        "mock": True,
    }


def get_weather(city: str) -> dict[str, Any]:
    return {
        "city": city,
        "temperature": 28,
        "unit": "celsius",
        "condition": "sunny",
        "mock": True,
    }


def get_order_status(order_id: str) -> dict[str, Any]:
    order = MOCK_ORDERS.get(order_id)
    if order is None:
        return {
            "order_id": order_id,
            "message": "No order is found with the provided order ID.",
            "found": False,
        }
    return {"found": True, **order, "mock": True}

def create_support_ticket(
    order_id: str,
    issue_description: str,
    database: MutableMapping[str, Any],
    user_id: str,
) -> dict[str, Any]:
    ticket_id = f"TICKET-{uuid4().hex[:12].upper()}"
    ticket_details = {
        "ticket_id": ticket_id,
        "order_id": order_id,
        "issue_description": issue_description,
        "status": "Open",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mock": True,
        "owner_user_id": user_id,
    }
    database[f"support_tickets/{ticket_id}"] = ticket_details
    return {key: value for key, value in ticket_details.items() if key != "owner_user_id"}


def request_refund(
    order_id: str,
    amount_cents: int,
    reason: str,
    pending_approvals: MutableMapping[str, Any],
    user_id: str,
) -> dict[str, Any]:
    approval_id = str(uuid4())
    approval = {
        "approval_id": approval_id,
        "action": "issue_mock_refund",
        "order_id": order_id,
        "amount_cents": amount_cents,
        "reason": reason,
        "approval_status": "pending",
        "owner_user_id": user_id,
    }
    pending_approvals[approval_id] = approval
    return {key: value for key, value in approval.items() if key != "owner_user_id"}


TOOL_HANDLERS: dict[str, tuple[type, Callable[..., dict[str, Any]]]] = {
    "get_weather": (WeatherArguments, get_weather),
    "get_order_status": (OrderStatusArguments, get_order_status),
    "create_support_ticket": (SupportTicketArguments, create_support_ticket),
    "get_support_ticket_details": (SupportTicketDetailsArguments, get_support_ticket_details),
    "request_refund": (RefundApprovalArguments, request_refund),
}


class InvalidToolCall(Exception):
    pass


class MissingDemoUserIdentity(Exception):
    pass


def normalize_demo_user_id(user_id: str | None) -> str | None:
    try:
        return str(UUID(user_id)) if user_id is not None else None
    except ValueError:
        return None


def map_tool_calls(
    tool_calls: list[Any],
    database: MutableMapping[str, Any],
    user_id: str | None = None,
    pending_approvals: MutableMapping[str, Any] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    user_scoped_tool_names = {
        "create_support_ticket",
        "get_support_ticket_details",
        "request_refund",
    }
    normalized_user_id = normalize_demo_user_id(user_id)
    if any(call.function.name in user_scoped_tool_names for call in tool_calls):
        if normalized_user_id is None:
            raise MissingDemoUserIdentity

    tool_messages = []
    executions = []
    for tool_call in tool_calls:
        tool = TOOL_HANDLERS.get(tool_call.function.name)
        if tool is None:
            raise InvalidToolCall(f"Unregistered tool: {tool_call.function.name}")
        argument_model, handler = tool
        try:
            arguments = json.loads(tool_call.function.arguments)
            validated_arguments = argument_model.model_validate(arguments)
        except (json.JSONDecodeError, TypeError, ValidationError) as exc:
            raise InvalidToolCall("Tool arguments did not match the registered schema") from exc
        arguments = validated_arguments.model_dump()
        if tool_call.function.name in user_scoped_tool_names:
            if normalized_user_id is None:
                raise MissingDemoUserIdentity
            if tool_call.function.name == "request_refund":
                if pending_approvals is None:
                    raise InvalidToolCall("Pending-approval storage is not configured")
                result = handler(
                    **arguments,
                    pending_approvals=pending_approvals,
                    user_id=normalized_user_id,
                )
            else:
                result = handler(**arguments, database=database, user_id=normalized_user_id)
        else:
            result = handler(**arguments)
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })
        executions.append({
            "name": tool_call.function.name,
            "arguments": validated_arguments.model_dump(),
            "result": result,
        })
    return tool_messages, executions


def _provider_shape_error(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "error": "The provider returned an invalid tool-calling response.",
            "request_id": request.state.request_id,
        },
        status_code=502,
    )
