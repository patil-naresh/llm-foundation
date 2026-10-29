from typing import Any

from fastapi import Request


def client_for_model(request: Request, model: str) -> Any:
    config = request.app.state.config_values
    provider = config["MODEL_PROVIDER_MAP"].get(model, "DEFAULT")
    return request.app.state.provider_clients[provider]
