"""Dataset endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.schemas import CO2Dataset
from services.dataset_cache import get_co2_dataframe

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("/co2", response_model=CO2Dataset)
def get_co2(force_refresh: bool = Query(False)) -> CO2Dataset:
    try:
        df, source = get_co2_dataframe(force_refresh=force_refresh)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"failed to load CO2 data: {e}")

    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    values = df["co2"].astype(float).tolist()
    return CO2Dataset(
        source=source,
        n_records=len(df),
        start_date=dates[0],
        end_date=dates[-1],
        dates=dates,
        values=values,
    )
