from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from app.services.data_services import load_data
from app.config import DATA_PATH

router = APIRouter()



@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_name = file.filename

    if not file_name:
        raise HTTPException(
            status_code=400, detail="Uploaded file must include a filename"
        )

    file_name = os.path.basename(file_name)
    file_path = os.path.join(DATA_PATH, file_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    sample_data = None
    file_columns = None

    data = load_data(file_path)

    sample_data = data.head(5).to_dict(orient="records")
    file_columns = data.columns.to_list()

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "columns" : file_columns,
        "headRows": sample_data,
    }

# Route for getting Names
@router.get('/upload/get-files')
async def get_upoads():
    if not os.path.isdir(DATA_PATH):
        raise HTTPException(
            status_code=404, detail=f"Raw data directory not found"
        )

    file_names = sorted(
        [
            entry
            for entry in os.listdir(DATA_PATH)
            if os.path.isfile(os.path.join(DATA_PATH, entry))
        ]
    )

    return {"files": file_names}