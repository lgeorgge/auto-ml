from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier


def get_classification_models():
    return{
        "logistic_regression" : LogisticRegression(max_iter=1000),
        "random_forest" : RandomForestClassifier()
        
    }

def get_regression_models():
    return{
        'linear_regression' : LinearRegression()

    }