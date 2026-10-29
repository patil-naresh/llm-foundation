from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

mock_db_api = APIRouter(prefix="/v1/mock-db", tags=["mock database"])


class MockDatabaseEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    value: Any


@mock_db_api.get("")
async def get_entries(request: Request, key: str | None = Query(default=None)):
    entries: dict[str, Any] = request.app.state.mock_database
    if key is not None:
        if key not in entries:
            raise HTTPException(status_code=404, detail="Mock database key not found")
        return {"key": key, "value": entries[key]}
    return {"items": entries.copy()}


@mock_db_api.put("")
async def set_entry(payload: MockDatabaseEntry, request: Request):
    if not payload.key.strip():
        raise HTTPException(status_code=422, detail="Key must not be empty")
    request.app.state.mock_database[payload.key] = payload.value
    return {"key": payload.key, "value": payload.value}


@mock_db_api.delete("")
async def delete_entry(request: Request, key: str | None = Query(default=None)):
    entries: dict[str, Any] = request.app.state.mock_database
    if key is None:
        deleted_count = len(entries)
        entries.clear()
        return {"deleted_count": deleted_count}
    if key not in entries:
        raise HTTPException(status_code=404, detail="Mock database key not found")
    del entries[key]
    return {"deleted": key}
