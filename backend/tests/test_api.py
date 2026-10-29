import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import app as app_module
from app import create_app

DEMO_USER_ID = "11111111-1111-4111-8111-111111111111"
OTHER_DEMO_USER_ID = "22222222-2222-4222-8222-222222222222"


def make_client():
    return SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock()),
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(), parse=AsyncMock())
        ),
        close=AsyncMock(),
    )


@pytest.fixture
def api_app():
    client = make_client()
    application = create_app(
        {
            "PROVIDER_API_KEY": "test-key-not-a-real-secret",
            "PROVIDER_BASE_URL": "https://provider.example/v1",
            "ALLOWED_MODELS": "model-a,model-b",
            "CORS_ORIGINS": "http://localhost:8080",
            "CONNECT_TIMEOUT_SECONDS": "1",
            "READ_TIMEOUT_SECONDS": "10",
        },
        openai_client=client,
    )
    return application, client


@pytest_asyncio.fixture
async def client(api_app):
    application, _ = api_app
    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test",
    ) as test_client:
        yield test_client


def completion_result(content="answer", model="model-a"):
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, parsed=None, tool_calls=None)
            )
        ],
        model_dump=lambda **_: {
            "id": "chatcmpl-test",
            "model": model,
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 4},
        },
    )


@pytest.mark.asyncio
async def test_health_and_model_allowlist(client):
    assert (await client.get("/health")).json() == {"status": "ok"}
    assert (await client.get("/v1/models")).json() == {
        "object": "list",
        "data": [
            {"id": "model-a", "object": "model"},
            {"id": "model-b", "object": "model"},
        ],
    }


@pytest.mark.asyncio
async def test_cors_allows_demo_user_id_header(client):
    response = await client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-demo-user-id",
        },
    )

    assert response.status_code == 200
    assert "x-demo-user-id" in response.headers["access-control-allow-headers"].lower()


@pytest.mark.asyncio
async def test_mock_database_stores_reads_lists_and_deletes_arbitrary_json(client):
    value = {
        "name": "Sample",
        "settings": {"enabled": True, "threshold": 0.75},
        "tags": ["mock", "test"],
        "optional": None,
    }

    saved = await client.put(
        "/v1/mock-db",
        json={"key": "profile:sample", "value": value},
    )
    assert saved.status_code == 200
    assert saved.json() == {"key": "profile:sample", "value": value}

    loaded = await client.get("/v1/mock-db", params={"key": "profile:sample"})
    assert loaded.status_code == 200
    assert loaded.json() == {"key": "profile:sample", "value": value}

    listed = await client.get("/v1/mock-db")
    assert listed.status_code == 200
    assert listed.json() == {"items": {"profile:sample": value}}

    deleted = await client.delete("/v1/mock-db", params={"key": "profile:sample"})
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": "profile:sample"}
    assert (await client.get("/v1/mock-db")).json() == {"items": {}}


@pytest.mark.asyncio
async def test_mock_database_overwrites_values_and_clears_all_entries(client):
    await client.put("/v1/mock-db", json={"key": "counter", "value": 1})
    overwritten = await client.put("/v1/mock-db", json={"key": "counter", "value": 2})
    await client.put("/v1/mock-db", json={"key": "enabled", "value": True})

    assert overwritten.json()["value"] == 2
    cleared = await client.delete("/v1/mock-db")
    assert cleared.status_code == 200
    assert cleared.json() == {"deleted_count": 2}
    assert (await client.get("/v1/mock-db")).json() == {"items": {}}


@pytest.mark.asyncio
async def test_mock_database_missing_key_and_empty_key_errors(client):
    missing = await client.get("/v1/mock-db", params={"key": "missing"})
    assert missing.status_code == 404

    empty_key = await client.put("/v1/mock-db", json={"key": " ", "value": "value"})
    assert empty_key.status_code == 422


@pytest.mark.asyncio
async def test_mock_database_allows_cors_preflight_for_put(client):
    response = await client.options(
        "/v1/mock-db",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert "PUT" in response.headers["Access-Control-Allow-Methods"]


@pytest.mark.asyncio
async def test_frontend_streaming_contract_and_provider_options(client, api_app):
    _, provider = api_app
    provider.chat.completions.create.return_value = completion_result("Hello 🌍")

    response = await client.post(
        "/v1/chat/completions",
        headers={"X-Request-ID": "frontend-test"},
        json={
            "model": "model-a",
            "messages": [
                {"role": "user", "content": "Hello", "extra": "not-forwarded"},
                {"role": "assistant", "content": "Hi"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-Accel-Buffering"] == "no"
    assert response.headers["X-Request-ID"] == "frontend-test"
    assert '"content": "Hello 🌍"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")
    kwargs = provider.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is False
    assert kwargs["model"] == "model-a"
    assert [tool["function"]["name"] for tool in kwargs["tools"]] == [
        "get_weather",
        "get_order_status",
        "create_support_ticket",
        "get_support_ticket_details",
        "request_refund",
    ]
    assert kwargs["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]


@pytest.mark.asyncio
async def test_chat_completion_nonstream_supports_temperature_and_max_tokens(client, api_app):
    _, provider = api_app
    provider.chat.completions.create.return_value = completion_result()

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "model-b",
            "messages": [{"role": "user", "content": "Question"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "answer"
    kwargs = provider.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "model-b"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 100
    assert kwargs["stream"] is False


@pytest.mark.asyncio
async def test_chat_completion_forwards_additional_openai_parameters(client, api_app):
    _, provider = api_app
    provider.chat.completions.create.return_value = completion_result()

    response = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Question"}],
            "stream": False,
            "top_p": 0.8,
            "presence_penalty": 0.5,
            "stop": ["END"],
            "max_completion_tokens": 200,
        },
    )

    assert response.status_code == 200
    kwargs = provider.chat.completions.create.call_args.kwargs
    assert kwargs["top_p"] == 0.8
    assert kwargs["presence_penalty"] == 0.5
    assert kwargs["stop"] == ["END"]
    assert kwargs["max_completion_tokens"] == 200


@pytest.mark.asyncio
async def test_chat_completion_rejects_client_credentials_in_extra_parameters(client, api_app):
    _, provider = api_app
    response = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Question"}],
            "api_key": "must-not-be-forwarded",
        },
    )

    assert response.status_code == 400
    provider.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_chat_completion_routes_model_to_configured_provider(monkeypatch):
    monkeypatch.setenv("ALT_API_KEY", "alt-test-key")
    monkeypatch.setenv("ALT_BASE_URL", "https://alt-provider.example/v1")
    primary = make_client()
    secondary = make_client()
    secondary.chat.completions.create.return_value = completion_result(model="model-b")
    application = create_app(
        {
            "PROVIDER_API_KEY": "primary-test-key",
            "PROVIDER_BASE_URL": "https://primary-provider.example/v1",
            "ALLOWED_MODELS": "model-a,model-b",
            "MODEL_PROVIDER_MAP": '{"model-b":"ALT"}',
            "CORS_ORIGINS": "http://localhost:8080",
        },
        openai_client=primary,
    )
    application.state.provider_clients["ALT"] = secondary

    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test",
    ) as test_client:
        response = await test_client.post(
            "/v1/chat/completions",
            json={
                "model": "model-b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    secondary.chat.completions.create.assert_awaited_once()
    primary.chat.completions.create.assert_not_called()
    assert secondary.chat.completions.create.call_args.kwargs["model"] == "model-b"


@pytest.mark.asyncio
async def test_chat_completion_maps_registered_tool_calls_and_returns_final_answer(client, api_app):
    _, provider = api_app
    tool_call = SimpleNamespace(
        id="call-order",
        function=SimpleNamespace(name="get_order_status", arguments='{"order_id":"A123"}'),
    )
    assistant_message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        model_dump=lambda **_: {
            "role": "assistant",
            "tool_calls": [{"id": "call-order", "type": "function"}],
        },
    )
    provider.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[SimpleNamespace(message=assistant_message)],
            model="model-a",
        ),
        completion_result("The order has shipped."),
    ]

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "model-a",
            "messages": [{"role": "user", "content": "Check order A123"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "The order has shipped."
    assert response.json()["tool_results"] == [{
        "name": "get_order_status",
        "arguments": {"order_id": "A123"},
        "result": {
            "found": True,
            "order_id": "A123",
            "status": "Shipped",
            "estimated_delivery": "2026-10-29",
            "mock": True,
        },
    }]
    assert provider.chat.completions.create.await_count == 2
    first_call = provider.chat.completions.create.call_args_list[0].kwargs
    follow_up = provider.chat.completions.create.call_args_list[1].kwargs
    assert first_call["model"] == "model-a"
    assert first_call["stream"] is False
    assert follow_up["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-order",
        "content": (
            '{"found": true, "order_id": "A123", "status": "Shipped", '
            '"estimated_delivery": "2026-10-29", "mock": true}'
        ),
    }


@pytest.mark.asyncio
async def test_chat_completion_stream_includes_executed_tool_results(client, api_app):
    _, provider = api_app
    tool_call = SimpleNamespace(
        id="call-weather",
        function=SimpleNamespace(name="get_weather", arguments='{"city":"Delhi"}'),
    )
    assistant_message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        model_dump=lambda **_: {
            "role": "assistant",
            "tool_calls": [{"id": "call-weather", "type": "function"}],
        },
    )
    provider.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=assistant_message)], model="model-a"),
        completion_result("It is sunny in Delhi."),
    ]

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "model-a",
            "messages": [{"role": "user", "content": "What's the weather in Delhi?"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    events = [
        json.loads(line.removeprefix("data: ").strip())
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    assert events[0]["tool_results"][0] == {
        "name": "get_weather",
        "arguments": {"city": "Delhi"},
        "result": {
            "city": "Delhi",
            "temperature": 28,
            "unit": "celsius",
            "condition": "sunny",
            "mock": True,
        },
    }


@pytest.mark.asyncio
async def test_chat_completion_rejects_unknown_model(client, api_app):
    _, provider = api_app
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "not-allowed", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 400
    provider.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_messages_do_not_call_provider(client, api_app):
    _, provider = api_app
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "system", "content": " "}]},
    )
    assert response.status_code == 400
    provider.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_cors_preflight_and_untrusted_origin(client, api_app):
    _, provider = api_app
    preflight = await client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["Access-Control-Allow-Origin"] == "http://localhost:8080"
    assert "POST" in preflight.headers["Access-Control-Allow-Methods"]

    denied = await client.post(
        "/v1/chat/completions",
        headers={"Origin": "https://untrusted.example"},
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert denied.status_code == 403
    provider.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_request_body_size_limit(api_app):
    application, _ = api_app
    application.state.config_values["MAX_REQUEST_BYTES"] = 10
    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test",
    ) as test_client:
        response = await test_client.post("/v1/chat/completions", json={"query": "this is too large"})
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_chunked_request_body_size_limit(api_app):
    application, provider = api_app
    application.state.config_values["MAX_REQUEST_BYTES"] = 10

    async def oversized_body():
        yield b'{"query":'
        yield b'"too large"}'

    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test",
    ) as test_client:
        response = await test_client.post("/v1/chat/completions", content=oversized_body())

    assert response.status_code == 413
    provider.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_existing_single_query_responses_endpoint(client, api_app):
    _, provider = api_app
    provider.responses.create.return_value = SimpleNamespace(
        output_text="Hello",
        model="model-a",
        usage=SimpleNamespace(input_tokens=2, output_tokens=3, total_tokens=5),
    )

    response = await client.post("/v1/chat", json={"query": "Hello"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Hello"
    assert response.json()["usage"] == {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
    }
    assert provider.responses.create.call_args.kwargs["model"] == "model-a"


@pytest.mark.asyncio
async def test_json_prompt_scenario(client, api_app):
    _, provider = api_app
    provider.chat.completions.create.return_value = completion_result('{"name":"Sam"}')

    response = await client.post("/v1/structured-outputs/json-prompt", json={})

    assert response.status_code == 200
    assert response.json()["mode"] == "json-prompt"
    assert "Respond in JSON format." in (
        provider.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    )


@pytest.mark.asyncio
async def test_person_structured_output(client, api_app):
    _, provider = api_app
    person = SimpleNamespace(
        model_dump=lambda: {"name": "Sam", "age": 30, "city": "Delhi"}
    )
    provider.chat.completions.parse.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=person))],
        model="model-a",
    )

    response = await client.post("/v1/structured-outputs/person", json={})

    assert response.status_code == 200
    assert response.json()["person"] == {"name": "Sam", "age": 30, "city": "Delhi"}
    assert provider.chat.completions.parse.call_args.kwargs["response_format"].__name__ == "Person"


@pytest.mark.asyncio
async def test_order_status_tool_roundtrip(client, api_app):
    _, provider = api_app
    tool_call = SimpleNamespace(
        id="call-order",
        function=SimpleNamespace(name="get_order_status", arguments='{"order_id":"A123"}'),
    )
    assistant_message = SimpleNamespace(
        tool_calls=[tool_call],
        content=None,
        model_dump=lambda **_: {
            "role": "assistant",
            "tool_calls": [{"id": "call-order", "type": "function"}],
        },
    )
    provider.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=assistant_message)], model="model-a"),
        completion_result("Shipped, due Oct 29."),
    ]

    response = await client.post("/v1/tools", json={})

    assert response.status_code == 200
    assert response.json()["answer"] == "Shipped, due Oct 29."
    assert response.json()["tool_calls"][0]["name"] == "get_order_status"
    assert response.json()["tool_calls"][0]["result"]["status"] == "Shipped"
    follow_up = provider.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert follow_up[-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_support_ticket_tool_persists_ticket_in_mock_database(client, api_app):
    _, provider = api_app
    tool_call = SimpleNamespace(
        id="call-ticket",
        function=SimpleNamespace(
            name="create_support_ticket",
            arguments='{"order_id":"A123","issue_description":"Package arrived damaged"}',
        ),
    )
    assistant_message = SimpleNamespace(
        tool_calls=[tool_call],
        content=None,
        model_dump=lambda **_: {
            "role": "assistant",
            "tool_calls": [{"id": "call-ticket", "type": "function"}],
        },
    )
    provider.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=assistant_message)], model="model-a"),
        completion_result("I created a support ticket for the damaged package."),
    ]

    response = await client.post(
        "/v1/tools",
        headers={"X-Demo-User-ID": DEMO_USER_ID},
        json={"messages": [{"role": "user", "content": "Create a ticket for A123"}]},
    )

    assert response.status_code == 200
    ticket = response.json()["tool_calls"][0]["result"]
    stored = await client.get(
        "/v1/mock-db",
        params={"key": f"support_tickets/{ticket['ticket_id']}"},
    )
    assert stored.status_code == 200
    assert {
        key: value
        for key, value in stored.json()["value"].items()
        if key != "owner_user_id"
    } == ticket
    assert stored.json()["value"]["owner_user_id"] == DEMO_USER_ID
    assert ticket["order_id"] == "A123"
    assert ticket["issue_description"] == "Package arrived damaged"
    assert ticket["status"] == "Open"
    assert ticket["created_at"]


@pytest.mark.asyncio
async def test_chat_completion_support_ticket_tool_persists_ticket(client, api_app):
    _, provider = api_app
    tool_call = SimpleNamespace(
        id="call-ticket",
        function=SimpleNamespace(
            name="create_support_ticket",
            arguments='{"order_id":"B456","issue_description":"Delivery is delayed"}',
        ),
    )
    assistant_message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        model_dump=lambda **_: {
            "role": "assistant",
            "tool_calls": [{"id": "call-ticket", "type": "function"}],
        },
    )
    provider.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=assistant_message)], model="model-a"),
        completion_result("The delivery-delay ticket has been created."),
    ]

    response = await client.post(
        "/v1/chat/completions",
        headers={"X-Demo-User-ID": DEMO_USER_ID},
        json={
            "messages": [{"role": "user", "content": "Create a ticket for delayed order B456"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    stored_items = (await client.get("/v1/mock-db")).json()["items"]
    assert len(stored_items) == 1
    ticket = next(iter(stored_items.values()))
    assert ticket["order_id"] == "B456"
    assert ticket["issue_description"] == "Delivery is delayed"
    assert ticket["owner_user_id"] == DEMO_USER_ID


@pytest.mark.asyncio
async def test_refund_tool_requires_human_approval_before_issuing_refund(client, api_app):
    application, provider = api_app
    tool_call = SimpleNamespace(
        id="call-refund",
        function=SimpleNamespace(
            name="request_refund",
            arguments='{"order_id":"A123","amount_cents":1250,"reason":"Package arrived damaged"}',
        ),
    )
    assistant_message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        model_dump=lambda **_: {
            "role": "assistant",
            "tool_calls": [{"id": "call-refund", "type": "function"}],
        },
    )
    provider.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=assistant_message)], model="model-a"),
        completion_result("I requested approval for this refund."),
    ]

    response = await client.post(
        "/v1/chat/completions",
        headers={"X-Demo-User-ID": DEMO_USER_ID},
        json={
            "messages": [{"role": "user", "content": "Refund $12.50 for damaged goods"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    pending = response.json()["tool_results"][0]["result"]
    assert pending["approval_status"] == "pending"
    assert pending["amount_cents"] == 1250
    assert application.state.mock_database == {}

    approval_id = pending["approval_id"]
    other_user_response = await client.post(
        f"/v1/approvals/{approval_id}",
        headers={"X-Demo-User-ID": OTHER_DEMO_USER_ID},
        json={"approved": True},
    )
    assert other_user_response.status_code == 404
    assert application.state.mock_database == {}

    approval_response = await client.post(
        f"/v1/approvals/{approval_id}",
        headers={"X-Demo-User-ID": DEMO_USER_ID},
        json={"approved": True},
    )
    assert approval_response.status_code == 200
    assert approval_response.json()["approval_status"] == "approved"
    refund = approval_response.json()["result"]
    assert refund["status"] == "issued"
    assert refund["amount_cents"] == 1250
    assert application.state.mock_database[f"refunds/{refund['refund_id']}"] == refund

    duplicate_response = await client.post(
        f"/v1/approvals/{approval_id}",
        headers={"X-Demo-User-ID": DEMO_USER_ID},
        json={"approved": True},
    )
    assert duplicate_response.status_code == 409
    assert len(application.state.mock_database) == 1


@pytest.mark.asyncio
async def test_rejecting_refund_approval_does_not_issue_refund(client, api_app):
    application, _ = api_app
    approval_id = "demo-approval"
    application.state.pending_approvals[approval_id] = {
        "approval_id": approval_id,
        "action": "issue_mock_refund",
        "order_id": "A123",
        "amount_cents": 500,
        "reason": "Test rejection",
        "approval_status": "pending",
        "owner_user_id": DEMO_USER_ID,
    }

    response = await client.post(
        f"/v1/approvals/{approval_id}",
        headers={"X-Demo-User-ID": DEMO_USER_ID},
        json={"approved": False},
    )

    assert response.status_code == 200
    assert response.json()["approval_status"] == "rejected"
    assert application.state.mock_database == {}
    assert application.state.pending_approvals[approval_id]["approval_status"] == "rejected"


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/tools", "/v1/chat/completions"])
async def test_support_ticket_details_are_only_returned_to_the_ticket_owner(
    client, api_app, endpoint
):
    application, provider = api_app
    application.state.mock_database["support_tickets/TICKET-DEMO"] = {
        "ticket_id": "TICKET-DEMO",
        "order_id": "A123",
        "issue_description": "Package arrived damaged",
        "status": "Open",
        "owner_user_id": DEMO_USER_ID,
    }
    tool_call = SimpleNamespace(
        id="call-ticket-details",
        function=SimpleNamespace(
            name="get_support_ticket_details",
            arguments='{"ticket_id":"TICKET-DEMO"}',
        ),
    )
    assistant_message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        model_dump=lambda **_: {
            "role": "assistant",
            "tool_calls": [{"id": "call-ticket-details", "type": "function"}],
        },
    )
    provider.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=assistant_message)], model="model-a"),
        completion_result(),
    ]
    request_json = (
        {"messages": [{"role": "user", "content": "Show me ticket TICKET-DEMO"}]}
        if endpoint == "/v1/tools"
        else {
            "messages": [{"role": "user", "content": "Show me ticket TICKET-DEMO"}],
            "stream": False,
        }
    )

    owner_response = await client.post(
        endpoint,
        headers={"X-Demo-User-ID": DEMO_USER_ID},
        json=request_json,
    )

    assert owner_response.status_code == 200
    owner_result = (
        owner_response.json()["tool_calls"][0]["result"]
        if endpoint == "/v1/tools"
        else owner_response.json()["tool_results"][0]["result"]
    )
    assert owner_result["found"] is True
    assert "owner_user_id" not in owner_result

    provider.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=assistant_message)], model="model-a"),
        completion_result(),
    ]
    other_user_response = await client.post(
        endpoint,
        headers={"X-Demo-User-ID": OTHER_DEMO_USER_ID},
        json=request_json,
    )

    assert other_user_response.status_code == 200
    other_user_result = (
        other_user_response.json()["tool_calls"][0]["result"]
        if endpoint == "/v1/tools"
        else other_user_response.json()["tool_results"][0]["result"]
    )
    assert other_user_result["found"] is False
    assert "owner_user_id" not in other_user_result


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/tools", "/v1/chat/completions"])
@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Demo-User-ID": "not-a-uuid"}],
)
async def test_support_ticket_tools_require_a_valid_demo_user_id(
    client, api_app, endpoint, headers
):
    _, provider = api_app
    tool_call = SimpleNamespace(
        id="call-ticket-details",
        function=SimpleNamespace(
            name="get_support_ticket_details",
            arguments='{"ticket_id":"TICKET-DEMO"}',
        ),
    )
    assistant_message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        model_dump=lambda **_: {
            "role": "assistant",
            "tool_calls": [{"id": "call-ticket-details", "type": "function"}],
        },
    )
    provider.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=assistant_message)],
        model="model-a",
    )
    request_json = (
        {"messages": [{"role": "user", "content": "Show me ticket TICKET-DEMO"}]}
        if endpoint == "/v1/tools"
        else {
            "messages": [{"role": "user", "content": "Show me ticket TICKET-DEMO"}],
            "stream": False,
        }
    )

    response = await client.post(endpoint, headers=headers, json=request_json)

    assert response.status_code == 401
    assert "demo user ID" in response.json()["error"]
    assert provider.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_weather_tool_returns_mock_data(client, api_app):
    _, provider = api_app
    call = SimpleNamespace(
        id="call-weather",
        function=SimpleNamespace(name="get_weather", arguments='{"city":"Delhi"}'),
    )
    provider.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[call],
                        content=None,
                        model_dump=lambda **_: {"role": "assistant", "tool_calls": []},
                    )
                )
            ],
            model="model-a",
        ),
        completion_result("Sunny in Delhi."),
    ]

    response = await client.post("/v1/tools", json={})

    assert response.status_code == 200
    weather = response.json()["tool_calls"][0]["result"]
    assert weather["city"] == "Delhi"
    assert weather["mock"] is True


def test_invalid_config_fails_fast():
    with pytest.raises(ValueError, match="ALLOWED_MODELS"):
        create_app({"ALLOWED_MODELS": "  "}, openai_client=make_client())

    with pytest.raises(ValueError, match="CORS_ORIGINS"):
        create_app({"CORS_ORIGINS": "*"}, openai_client=make_client())

    with pytest.raises(ValueError, match="MODEL_PROVIDER_MAP"):
        create_app(
            {"MODEL_PROVIDER_MAP": '{"not-allowlisted":"OPENAI"}'},
            openai_client=make_client(),
        )


@pytest.mark.asyncio
async def test_lifespan_builds_httpx_timeout_with_default_and_connect(monkeypatch):
    created = []
    fake_clients = []
    monkeypatch.setenv("ALT_API_KEY", "alt-test-key")
    monkeypatch.setenv("ALT_BASE_URL", "https://alt-provider.example/v1")

    def fake_openai_client(**kwargs):
        created.append(kwargs)
        fake_client = make_client()
        fake_clients.append(fake_client)
        return fake_client

    monkeypatch.setattr(app_module, "AsyncOpenAI", fake_openai_client)
    application = create_app(
        {
            "PROVIDER_API_KEY": "test-key",
            "PROVIDER_BASE_URL": "https://provider.example/v1",
            "ALLOWED_MODELS": "model-a,model-b",
            "MODEL_PROVIDER_MAP": '{"model-b":"ALT"}',
            "CORS_ORIGINS": "http://localhost:8080",
            "CONNECT_TIMEOUT_SECONDS": "1",
            "READ_TIMEOUT_SECONDS": "10",
        }
    )

    async with application.router.lifespan_context(application):
        assert len(created) == 2
        by_key = {kwargs["api_key"]: kwargs for kwargs in created}
        assert by_key["test-key"]["base_url"] == "https://provider.example/v1"
        assert by_key["alt-test-key"]["base_url"] == "https://alt-provider.example/v1"
        timeout = created[0]["timeout"]
        assert timeout.connect == 1
        assert timeout.read == 10
        assert timeout.write == 10
        assert timeout.pool == 10

    for fake_client in fake_clients:
        fake_client.close.assert_awaited_once()
