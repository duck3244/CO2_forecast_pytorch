"""Model catalog endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from api.schemas import ModelsResponse
from services import model_registry

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    device, device_name = model_registry.device_info()
    return ModelsResponse(
        models=model_registry.list_models(),
        device=device,
        device_name=device_name,
    )
