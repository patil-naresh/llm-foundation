import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from dotenv import load_dotenv


def positive(value, name, convert=float):
    try:
        result = convert(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if result <= 0 or result == float("inf") or result != result:
        raise ValueError(f"{name} must be positive")
    return result


def configure_app(app, config=None):
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    app.state.config_values = {
        "PROVIDER_API_KEY": (
            os.getenv("PROVIDER_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ),
        "PROVIDER_BASE_URL": os.getenv(
            "PROVIDER_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ),
        "ALLOWED_MODELS": os.getenv("ALLOWED_MODELS", "gpt-4o-mini"),
        # "OPENAI_MODEL": os.getenv("OPENAI_MODEL", ""),
        "MODEL_PROVIDER_MAP": os.getenv("MODEL_PROVIDER_MAP", "{}"),
        "SYSTEM_INSTRUCTIONS": os.getenv(
            "SYSTEM_INSTRUCTIONS",
            "You are a helpful assistant. Answer clearly and accurately.",
        ),
        "CORS_ORIGINS": os.getenv("CORS_ORIGINS", "http://localhost:8080"),
        "CONNECT_TIMEOUT_SECONDS": os.getenv("CONNECT_TIMEOUT_SECONDS", "10"),
        "READ_TIMEOUT_SECONDS": os.getenv("READ_TIMEOUT_SECONDS", "300"),
        "MAX_REQUEST_BYTES": os.getenv("MAX_REQUEST_BYTES", "1048576"),
        "MAX_QUERY_LENGTH": os.getenv("MAX_QUERY_LENGTH", "12000"),
    }
    if config:
        app.state.config_values.update(config)
    values = app.state.config_values

    key = values["PROVIDER_API_KEY"].strip()
    if key and ("\r" in key or "\n" in key):
        raise ValueError("Set a valid provider API key")

    base = values["PROVIDER_BASE_URL"].strip().rstrip("/")
    parsed_base = urlsplit(base)
    if (
        parsed_base.scheme not in {"http", "https"}
        or not parsed_base.hostname
        or parsed_base.username
        or parsed_base.password
        or parsed_base.query
        or parsed_base.fragment
    ):
        raise ValueError("PROVIDER_BASE_URL must be an HTTP(S) API root without credentials or query")
    if parsed_base.scheme == "http" and parsed_base.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Non-local providers must use HTTPS to protect the API key")

    raw_models = values["ALLOWED_MODELS"]
    models = list(dict.fromkeys(model.strip() for model in raw_models.split(",") if model.strip()))
    if not models:
        raise ValueError("ALLOWED_MODELS must contain at least one model")
    default_model = models[0]

    raw_provider_map = values["MODEL_PROVIDER_MAP"]
    if isinstance(raw_provider_map, str):
        try:
            model_provider_map = json.loads(raw_provider_map)
        except json.JSONDecodeError as exc:
            raise ValueError("MODEL_PROVIDER_MAP must be a JSON object") from exc
    else:
        model_provider_map = raw_provider_map
    if not isinstance(model_provider_map, dict):
        raise ValueError("MODEL_PROVIDER_MAP must be a JSON object")
    normalized_provider_map = {}
    for model, provider in model_provider_map.items():
        if model not in models:
            raise ValueError(f"MODEL_PROVIDER_MAP contains a model not in ALLOWED_MODELS: {model}")
        if not isinstance(provider, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", provider):
            raise ValueError(f"MODEL_PROVIDER_MAP has an invalid provider name for model {model}")
        normalized_provider_map[model] = provider.upper()

    origins = [origin.strip() for origin in values["CORS_ORIGINS"].split(",") if origin.strip()]
    if not origins or "*" in origins:
        raise ValueError("CORS_ORIGINS requires explicit origins, not a wildcard")
    for origin in origins:
        parsed_origin = urlsplit(origin)
        if (
            parsed_origin.scheme not in {"http", "https"}
            or not parsed_origin.hostname
            or parsed_origin.username
            or parsed_origin.password
            or parsed_origin.path
            or parsed_origin.query
            or parsed_origin.fragment
        ):
            raise ValueError("CORS_ORIGINS entries must be exact origins without paths")

    values["PROVIDER_API_KEY"] = key
    values["UPSTREAM_BASE_URL"] = base
    values["MODEL_ALLOWLIST"] = models
    values["DEFAULT_MODEL"] = default_model
    values["MODEL_PROVIDER_MAP"] = normalized_provider_map
    provider_names = set(normalized_provider_map.values())
    if len(normalized_provider_map) < len(models):
        provider_names.add("DEFAULT")
    provider_settings = {}
    for provider in provider_names:
        if provider == "DEFAULT":
            provider_key = key
            provider_base = base
        else:
            provider_key = os.getenv(f"{provider}_API_KEY", "").strip()
            provider_base = os.getenv(f"{provider}_BASE_URL", "")
            if provider == "OPENAI" and not provider_base:
                provider_base = "https://api.openai.com/v1"
            provider_base = provider_base.strip().rstrip("/")
            parsed_provider_base = urlsplit(provider_base)
            if (
                parsed_provider_base.scheme not in {"http", "https"}
                or not parsed_provider_base.hostname
                or parsed_provider_base.username
                or parsed_provider_base.password
                or parsed_provider_base.query
                or parsed_provider_base.fragment
            ):
                raise ValueError(
                    f"{provider}_BASE_URL must be an HTTP(S) API root without credentials or query"
                )
            if (
                parsed_provider_base.scheme == "http"
                and parsed_provider_base.hostname not in {"localhost", "127.0.0.1", "::1"}
            ):
                raise ValueError(f"Non-local {provider} providers must use HTTPS to protect the API key")
        if provider_key and ("\r" in provider_key or "\n" in provider_key):
            raise ValueError(f"Set a valid {provider}_API_KEY")
        provider_settings[provider] = {
            "api_key": provider_key,
            "base_url": provider_base,
        }
    values["PROVIDER_SETTINGS"] = provider_settings
    values["ALLOWED_ORIGINS"] = origins
    values["UPSTREAM_TIMEOUT"] = (
        positive(values["CONNECT_TIMEOUT_SECONDS"], "CONNECT_TIMEOUT_SECONDS"),
        positive(values["READ_TIMEOUT_SECONDS"], "READ_TIMEOUT_SECONDS"),
    )
    values["MAX_REQUEST_BYTES"] = positive(
        values["MAX_REQUEST_BYTES"], "MAX_REQUEST_BYTES", int
    )
    values["MAX_QUERY_LENGTH"] = positive(
        values["MAX_QUERY_LENGTH"], "MAX_QUERY_LENGTH", int
    )
