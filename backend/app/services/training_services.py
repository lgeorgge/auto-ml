
from pandas import DataFrame
from app.services.data_services import load_data
from app.ml.preprocessor import build_preprocessor


def train_model(file_name: str,target_variable :str | None=  None,task_type = 'CLASSIFICATION'):
    # Load data
    df : DataFrame = load_data(file_name)

    if task_type =='CLASSIFICATION' or task_type == 'REGRESSION':
        if target_variable is None:
            return {"error": "Target column is required for supervised learning"}

        if target_variable not in df.columns :
            return {"error": "Target column not found in dataset"}

        X = df.drop(columns=[target_variable])
        y = df[target_variable]

        # build the precprocessor
        preprocessor = build_preprocessor(X)

        X_processed = preprocessor.fit_transform(X)

        return {
          
            "message": "Preprocessing completed",
            "original_shape": X.shape,
            "processed_shape": X_processed.shape
        
        }
    else :

        preprocessor = build_preprocessor(df)

        X_processed = preprocessor.fit_transform(df)

        return {
            "message": "Preprocessing completed",
            "original_shape": df.shape,
            "processed_shape": X_processed.shape
        }
