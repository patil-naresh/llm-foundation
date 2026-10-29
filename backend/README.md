# Parley FastAPI backend

Stateless Python API for the Parley chat UI and the LLM examples. Conversations
remain in the browser; this service does not store them.

## Setup

From this directory:

```sh
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your server-side provider key. Start the
API with:

```sh
uvicorn app:app --host 127.0.0.1 --port 8000
```

Swagger UI is at `http://localhost:8000/docs`. `python app.py` is also supported.
Never commit `.env` or expose provider keys to the browser. Startup fails if
the provider key is missing or invalid configuration is supplied.

## Configuration

| Variable | Purpose / default |
| --- | --- |
| `PROVIDER_API_KEY` | Server-only API key; `NVIDIA_API_KEY` and `OPENAI_API_KEY` are accepted as fallbacks |
| `PROVIDER_BASE_URL` | OpenAI-compatible API root; `https://api.openai.com/v1` |
| `ALLOWED_MODELS` | Comma-separated provider model IDs; `gpt-4o-mini` |
| `OPENAI_MODEL` | Default model for examples; defaults to the first allowlisted model |
| `MODEL_PROVIDER_MAP` | Optional JSON object mapping model IDs to provider names; unmapped models use `PROVIDER_API_KEY` and `PROVIDER_BASE_URL` |
| `SYSTEM_INSTRUCTIONS` | Default assistant instructions |
| `CORS_ORIGINS` | Exact comma-separated browser origins, without paths; `http://localhost:8080` |
| `HOST`, `PORT` | Local listener; `127.0.0.1`, `8000` |
| `CONNECT_TIMEOUT_SECONDS`, `READ_TIMEOUT_SECONDS` | Provider connection/read timeouts; `10`, `300` |
| `MAX_REQUEST_BYTES` | Request size limit; `1048576` |
| `MAX_QUERY_LENGTH` | `/v1/chat` query length; `12000` |

For multiple providers, map each model ID to a provider name in
`MODEL_PROVIDER_MAP`, then set `<PROVIDER>_API_KEY` and
`<PROVIDER>_BASE_URL` for each mapped provider. Provider names are
case-insensitive and use letters, numbers, and underscores. For example:

```dotenv
ALLOWED_MODELS=gpt-4o-mini,meta/llama-3.1-70b-instruct
OPENAI_MODEL=gpt-4o-mini
MODEL_PROVIDER_MAP={"gpt-4o-mini":"OPENAI","meta/llama-3.1-70b-instruct":"NVIDIA"}
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
NVIDIA_API_KEY=...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

`OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`; other mapped
providers must specify their base URL. Models omitted from the map use the
legacy `PROVIDER_API_KEY` and `PROVIDER_BASE_URL` configuration. Each provider
gets its own client, and clients are selected by model without mutating shared
credentials. Only HTTPS provider URLs are allowed except loopback HTTP for local
development. CORS requires explicit origins; wildcards are rejected.

## Endpoints

- `GET /health` — service health (does not contact the provider).
- `GET /v1/models` — the configured model allowlist used by the UI.
- `GET /v1/mock-db` — list all in-memory mock database entries; pass `?key=...`
  to read a single entry.
- `PUT /v1/mock-db` — store or replace an entry using `{"key":"...","value":...}`.
- `DELETE /v1/mock-db?key=...` — delete one entry; `DELETE /v1/mock-db`
  clears all entries.
- `POST /v1/chat/completions` — compatible with the UI's OpenAI-style message
  history and SSE stream. Streaming defaults to enabled to preserve the UI
  contract; set `"stream": false` to receive the regular completion response.
  Supports `temperature`, `max_tokens`, and additional OpenAI-compatible chat
  completion parameters supplied in the request. The UI exposes temperature,
  max tokens, and a dynamic field editor for other JSON-valued parameters.
  The backend can execute the registered `get_weather` and `get_order_status`
  tools, return their results to the model, and send the resulting assistant
  answer back to the UI.
- `POST /v1/chat` — single-query Responses API example with optional
  `system_instructions` and token usage.
- `POST /v1/structured-outputs/plain` — ordinary text generation.
- `POST /v1/structured-outputs/json-prompt` — requests JSON in the prompt; this
  is a prompt suggestion, not schema enforcement.
- `POST /v1/structured-outputs/person` — Pydantic `Person` structured output.
  Provider/model support for structured outputs is required.
- `POST /v1/tools` — tool-calling demonstration with mock order-status and
  weather functions. The sample data is not live. Only the two registered tools
  can run; tool arguments are validated before execution.

Standalone versions of the learning examples are also in this directory:
`1_llm_basics.py`, `2_structured_outputs.py`, and `3_tool_calls.py`. Run them
from `backend/` after configuring the provider key in `.env`.

The mock database accepts arbitrary JSON values and is held in memory per app
instance. It starts empty and resets when the backend restarts. It is intended
for local development and synthetic test data only—not durable or access
controlled storage for real or sensitive data.
The `create_support_ticket` tool stores each created ticket at
`support_tickets/<ticket_id>` in this mock database. The frontend stores a
dummy UUID in localStorage and sends it as `X-Demo-User-ID`; ticket reads
through the support-ticket tool are restricted to tickets created with that
same ID. This is only a local demo ownership check: users can change the ID, so
do not use it as authentication or to protect real or sensitive data. The
mock-database endpoints remain unauthenticated and are for local testing only.
The `request_refund` tool demonstrates human approval: it creates a pending
mock request, and the chat UI asks for approval before `POST /v1/approvals/{id}`
records an issued mock refund. Rejecting the request does not issue a refund.
Approval requests and mock refunds are in-memory demo data, not a payment
integration or production approval system.

### Chat Completions example

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant who makes jokes."},
    {"role": "user", "content": "What is the square root of 49?"}
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

For conversation history, send all prior `user` and `assistant` messages in
order. The API key is never accepted from the request body.

## Security and operations

The API rejects browser origins outside the configured CORS allowlist, does not
follow provider redirects, validates models against the allowlist, and avoids
returning raw provider errors or credentials. CORS is not authentication or
rate-limiting; add caller authentication, quotas, and HTTPS before exposing a
paid API publicly. Provider data-retention policies still apply.

For local development, keep the service bound to `127.0.0.1`. For deployment,
run Uvicorn behind a TLS-terminating reverse proxy and set proxy/request
timeouts appropriately. Disable proxy buffering for SSE responses.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use mocked provider responses; they do not need a real API key or make
paid requests.
