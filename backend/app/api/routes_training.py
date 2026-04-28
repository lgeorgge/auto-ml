from fastapi import APIRouter, HTTPException

from app.services.training_services import train_model

router = APIRouter()

DATA_DIR = "data/raw"

@router.post("/train")
def train(filename: str, task_type: str, target_column: str | None = None):
        try:
            return train_model(filename, target_column, task_type)
        except ValueError:
              return HTTPException(400)
    
    