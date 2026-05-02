import logging
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import DATA_PATH
from app.services.training_services import train_model

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/train")
async def train(filename: str, task_type: str, target_column: str | None = None):
    try:
        if not filename or not filename.strip():
            raise HTTPException(status_code=400, detail="filename is required")

        safe_filename = os.path.basename(filename.strip())
        file_path = os.path.join(DATA_PATH, safe_filename)
        if not os.path.isfile(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found in raw data directory: {safe_filename}",
            )

        if not task_type or not task_type.strip():
            raise HTTPException(status_code=400, detail="task_type is required")

        normalized_task_type = task_type.strip().upper()
        allowed_task_types = {"CLASSIFICATION", "REGRESSION", "CLUSTERING"}
        if normalized_task_type not in allowed_task_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task_type '{task_type}'. Allowed values: {sorted(allowed_task_types)}",
            )

        result = train_model(safe_filename, target_column, normalized_task_type)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected training error")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/download/{task_type}")
async def download_model(task_type: str):
    allowed_task_types = {"CLASSIFICATION", "REGRESSION", "CLUSTERING"}
    normalized_task_type = task_type.strip().upper()
    if normalized_task_type not in allowed_task_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_type '{task_type}'. Allowed values: {sorted(allowed_task_types)}",
        )

    from app.config import MODEL_PATH
    task_dir = os.path.join(MODEL_PATH, normalized_task_type)
    
    if not os.path.exists(task_dir):
        raise HTTPException(status_code=404, detail=f"No models found for task {normalized_task_type}")
        
    for existing_file in os.listdir(task_dir):
        if existing_file.endswith(".joblib") or existing_file.endswith(".pkl"):
            return FileResponse(
                path=os.path.join(task_dir, existing_file),
                filename=existing_file,
                media_type="application/octet-stream"
            )
            
    raise HTTPException(status_code=404, detail="Model file not found")
