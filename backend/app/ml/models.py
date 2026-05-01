from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


def get_classification_models():
    return{
        "logistic_regression" : LogisticRegression(max_iter=1000),
        "random_forest" : RandomForestClassifier()
        
    }

def get_regression_models():
    return{
        'linear_regression' : LinearRegression()

    }
    
def get_cluster_model():
    return{
        'KMeans_Clustering' : KMeans()

    }
    
