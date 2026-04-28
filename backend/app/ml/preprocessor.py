


from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X : DataFrame):
    
    # First collect numerical and categorical data and split the features according to type
    numeric_features = X.select_dtypes(include=['int64','float64','float32','int32']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Pipelines
    # for numeric, replace with mean and scale
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="mean"),
             ("scaler",StandardScaler()))
            ]
    )

    # for categorical, impute and encode
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])


    # now preprocess
    preprocessor = ColumnTransformer(transformers=[
        ('numerical',numeric_pipeline,numeric_features),
        ('categorical',categorical_pipeline,categorical_features)
    ])

    return preprocessor
