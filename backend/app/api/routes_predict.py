#                                                       بسم الله الرحمن الرحيم                                                 //
# program: routes_predict.cpp 
# Description: 
# Author:  Abdallah Gasem
# Date: 01-05-2026
# Version: 1.0
# ----------------------------------------------------------------------------------------------------------------------------- //

from fastapi import APIRouter, HTTPException
from app.config import MODEL_PATH
import os
import joblib
import pandas as pd


router = APIRouter()

@router.post("/classify")
def predict_classification(payload: dict):
    model_dir_path = os.path.join(MODEL_PATH, "CLASSIFICATION")

    if not os.path.isdir(model_dir_path):
        raise HTTPException(status_code=404, detail="Classification model directory not found")

    model_files = [f for f in os.listdir(model_dir_path) if f.endswith(".joblib")]
    if not model_files:
        raise HTTPException(status_code=404, detail="No classification model found")

    model_path = os.path.join(model_dir_path, model_files[0])

    try:
        clf = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    features = payload.get("features", payload)

    if isinstance(features, dict):
        features_df = pd.DataFrame([features])
    elif isinstance(features, list) and len(features) > 0 and isinstance(features[0], dict):
        features_df = pd.DataFrame(features)

    try:
        prediction = clf.predict(features_df)
        return {"prediction": prediction.tolist() if hasattr(prediction, "tolist") else prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@router.post("/regress")
def predict_regression(payload: dict):
    model_dir_path = os.path.join(MODEL_PATH, "REGRESSION")

    if not os.path.isdir(model_dir_path):
        raise HTTPException(status_code=404, detail="REGRESSION model directory not found")

    model_files = [f for f in os.listdir(model_dir_path) if f.endswith(".joblib")]
    if not model_files:
        raise HTTPException(status_code=404, detail="No REGRESSION model found")

    model_path = os.path.join(model_dir_path, model_files[0])

    try:
        regressor = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    features = payload.get("features", payload)

    if isinstance(features, dict):
        features_df = pd.DataFrame([features])
    elif isinstance(features, list) and len(features) > 0 and isinstance(features[0], dict):
        # baatch predction with list of dicts
        features_df = pd.DataFrame(features)


    try:
        prediction = regressor.predict(features)
        return {"prediction": prediction.tolist() if hasattr(prediction, "tolist") else prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@router.post("/cluster")
def predict_cluster(payload: dict):
    model_dir_path = os.path.join(MODEL_PATH, "CLUSTERING")

    if not os.path.isdir(model_dir_path):
        raise HTTPException(status_code=404, detail="CLUSTERING model directory not found")

    model_files = [f for f in os.listdir(model_dir_path) if f.endswith(".joblib")]
    if not model_files:
        raise HTTPException(status_code=404, detail="No CLUSTERING model found")

    model_path = os.path.join(model_dir_path, model_files[0])

    try:
        clusterer = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    features = payload.get("features", payload)

    if isinstance(features, dict):
        features_df = pd.DataFrame([features])
    elif isinstance(features, list) and len(features) > 0 and isinstance(features[0], dict):
        features_df = pd.DataFrame(features)

    try:
        prediction = clusterer.predict(features_df)
        return {"prediction": prediction.tolist() if hasattr(prediction, "tolist") else prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Clustering prediction failed: {str(e)}")






















































