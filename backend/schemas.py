from typing import Literal

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=1000000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be empty")
        return value


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    stream: bool = True

    @model_validator(mode="before")
    @classmethod
    def reject_client_controlled_options(cls, value: Any) -> Any:
        if isinstance(value, dict):
            restricted = {
                "api_key",
                "base_url",
                "default_headers",
                "extra_body",
                "extra_headers",
                "extra_query",
                "http_client",
                "max_retries",
                "organization",
                "project",
                "timeout",
            }
            supplied = restricted.intersection(value)
            if supplied:
                names = ", ".join(sorted(supplied))
                raise ValueError(f"These request parameters are not allowed: {names}")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000000)
    system_instructions: str | None = Field(default=None, max_length=8000)


class StructuredOutputRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    query: str = Field(
        default="Give me the name, age and city of a fictional person.",
        min_length=1,
        max_length=1000000,
    )
    system_instructions: str | None = Field(default=None, max_length=8000)


class ToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    messages: list[ChatMessage] = Field(
        default_factory=lambda: [
            ChatMessage(
                role="user",
                content="Can you check order A123 and tell me its status and estimated delivery?",
            )
        ],
        min_length=1,
        max_length=100,
    )


class Person(BaseModel):
    name: str
    age: int
    city: str


class WeatherArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=1, max_length=200)


class OrderStatusArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(min_length=1, max_length=100)

class SupportTicketArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(min_length=1, max_length=100)
    issue_description: str = Field(min_length=1, max_length=1000)

class SupportTicketDetailsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticket_id: str = Field(min_length=1, max_length=100)


class RefundApprovalArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(min_length=1, max_length=100)
    amount_cents: int = Field(gt=0, le=100_000_000)
    reason: str = Field(min_length=1, max_length=500)


class ApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool