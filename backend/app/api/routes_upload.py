from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import pandas as pd

router = APIRouter()

UPLOAD_DIR = "data/raw"

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_string_path = file.filename
    
    if not file_string_path:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")

    file_string_path = os.path.basename(file_string_path)
    file_path = os.path.join(UPLOAD_DIR,file_string_path)


    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    

    sample_data = None

    if file:
        print(file.filename)
        print(file.size)
        print(file.headers)

    if file.filename and file.filename.endswith(".csv"):
        data = pd.read_csv(file_path)
        sample_data = data.head(5)
    else: 
        data = pd.read_excel(file.file)
        sample_data = data.head(5)

    return {"filename": file.filename, "message": "File uploaded successfully",
            "headRows" : sample_data }
