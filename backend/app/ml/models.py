from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering


def get_classification_models():
    return{
        "logistic_regression" : LogisticRegression(max_iter=1000),
        "random_forest" : RandomForestClassifier()
        
    }

def get_regression_models():
    return{
        'linear_regression' : LinearRegression(),
        "random_forest" : RandomForestRegressor()

    }
    
def get_cluster_model():
    return{
        'KMeans_Clustering' : KMeans(),
        'Agglomerative Clustering' : AgglomerativeClustering()

    }
    

