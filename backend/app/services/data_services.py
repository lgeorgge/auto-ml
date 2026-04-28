import os
import pandas as pd
from app.config import DATA_PATH

def load_data(file_name: str):
    df : pd.DataFrame
    file_path = os.path.join(DATA_PATH,os.path.basename(file_name))
    if file_path.endswith('.csv'):
        df =  pd.read_csv(file_path)
    elif file_path.endswith('xlsx'):
        df =  pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    if df.empty:
        raise FileNotFoundError("No file found")
    return df