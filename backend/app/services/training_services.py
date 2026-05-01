import os

import joblib
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from app.services.data_services import load_data
from app.ml.preprocessor import build_preprocessor
from app.ml.models import get_classification_models, get_cluster_model, get_regression_models
from app.ml.evaluation import evaluate_classification
from app.config import MODEL_PATH


def train_model(
    file_name: str, target_variable: str | None = None, task_type="CLASSIFICATION"
):
    # Load data
    df: DataFrame = load_data(file_name)

    # Remove ID columns
    df = df.drop(columns=[col for col in df.columns if "id" in col.lower()])

    try:

        if task_type.upper() == "CLASSIFICATION":
            if target_variable is None:
                raise ValueError(
                    "Target variable is required when using classification"
                )

            if target_variable not in df.columns:
                return {"error": "Target column not found in dataset"}

            X = df.drop(columns=[target_variable])
            y = df[target_variable]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Split

            models = get_classification_models()
            results = {}

            best_score = 0
            best_pipeline = None
            best_model_name = ""

            for name, model in models.items():
                print("started with model : ", name)

                # Build fresh preprocessor each time
                preprocessor = build_preprocessor(X_train)

                pipeline = Pipeline(
                    steps=[("preprocessor", preprocessor), ("model", model)]
                )

                pipeline.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = evaluate_classification(y_test, y_pred)
                results[name] = metrics

                # Select best based on F1-score
                if metrics["f1_score"] > best_score:
                    best_score = metrics["f1_score"]
                    best_pipeline = pipeline
                    best_model_name = name

            # updating the best model path to express the type of task
            task_dir = os.path.join(MODEL_PATH, task_type.upper())
            os.makedirs(task_dir, exist_ok=True)

            # Delete any existing model files in this directory first to make sure in the predict we are hitting the right model for the current task(bad design bas hanmashy 7alna!)
            for existing_file in os.listdir(task_dir):
                if existing_file.endswith(".joblib"):
                    existing_path = os.path.join(task_dir, existing_file)
                    if os.path.exists(existing_path):
                        os.remove(existing_path)

            model_path = os.path.join(task_dir, f"{best_model_name}.joblib")
            joblib.dump(best_pipeline, model_path)

            return {
                "message": "Training completed",
                "results": results,
                "best_model": best_model_name,
            }

        else:

            preprocessor = build_preprocessor(df)

            X_processed = preprocessor.fit_transform(df)

            return {
                "message": "Preprocessing completed",
                "original_shape": df.shape,
                "processed_shape": X_processed.shape,
            }
    except Exception as ex:
        print(str(ex))
        raise ex

def train_regression():
    model = get_regression_models()
    # ...continue ya wagdy!
    
def cluster_fit():
    model = get_cluster_model()
    # ...continue ya wagdy!
    