import streamlit as st
import requests
import pandas as pd
import io

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AutoML Platform", layout="wide")

st.title("AutoML Platform")
st.markdown("Upload raw data, select an ML task, and receive a fully trained and evaluated model without writing code.")

# Initialize session state variables
if "filename" not in st.session_state:
    st.session_state.filename = None
if "columns" not in st.session_state:
    st.session_state.columns = []
if "train_results" not in st.session_state:
    st.session_state.train_results = None
if "task_type" not in st.session_state:
    st.session_state.task_type = None

# 1. Data Ingestion
st.header("1. Upload Data")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Need to avoid re-uploading on every rerun
    if st.session_state.filename != uploaded_file.name:
        with st.spinner("Uploading data..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                response = requests.post(f"{API_URL}/upload", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("File uploaded successfully!")
                    st.session_state.filename = data.get("filename")
                    st.session_state.columns = data.get("columns", [])
                    
                    st.subheader("Data Preview")
                    df_preview = pd.DataFrame(data.get("headRows", []))
                    st.dataframe(df_preview)
                else:
                    st.error(f"Error uploading file: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
    else:
        st.success(f"File '{st.session_state.filename}' is ready.")

# 2. Task Selection
st.header("2. Task Selection")
task_type = st.selectbox("Select ML Problem Type", ["Classification", "Regression", "Clustering"])

target_column = None
if task_type in ["Classification", "Regression"]:
    if st.session_state.columns:
        target_column = st.selectbox("Select Target Variable (Label)", st.session_state.columns)
    else:
        st.info("Please upload a dataset first to select the target variable.")

# 3. Model Training
st.header("3. Train Model")
if st.button("Train Model", type="primary", disabled=(not st.session_state.filename)):
    st.session_state.task_type = task_type
    with st.spinner("Training models... This might take a while."):
        try:
            params = {
                "filename": st.session_state.filename,
                "task_type": task_type.upper()
            }
            if target_column:
                params["target_column"] = target_column
                
            response = requests.post(f"{API_URL}/train", params=params)
            
            if response.status_code == 200:
                st.session_state.train_results = response.json()
                st.success("Training completed successfully!")
            else:
                st.error(f"Training failed: {response.text}")
                st.session_state.train_results = None
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")

# 4. Results & Export
if st.session_state.train_results:
    st.header("4. Results & Export")
    results = st.session_state.train_results
    
    st.subheader("Model Performance")
    
    if "best_model" in results:
        st.info(f"**Best Model Selected:** {results['best_model']}")
        
    if "results" in results:
        for model_name, metrics in results["results"].items():
            with st.expander(f"Metrics for {model_name}", expanded=(model_name == results.get("best_model"))):
                # Display metrics neatly
                cols = st.columns(len(metrics))
                for idx, (metric_name, metric_value) in enumerate(metrics.items()):
                    if isinstance(metric_value, (int, float)):
                        cols[idx].metric(label=metric_name.replace("_", " ").title(), value=f"{metric_value:.4f}")
                    elif metric_name == "confusion_matrix":
                        st.write("Confusion Matrix:")
                        st.dataframe(pd.DataFrame(metric_value))
                    else:
                        st.write(f"{metric_name.replace('_', ' ').title()}: {metric_value}")
    else:
        st.json(results)
    
    st.subheader("Export Model")
    st.write("Download the best trained model and preprocessing pipeline.")
    
    try:
        # Fetch the model directly from the backend
        task_str = st.session_state.task_type.upper()
        download_response = requests.get(f"{API_URL}/download/{task_str}")
        if download_response.status_code == 200:
            st.download_button(
                label="Download Model (.joblib)",
                data=download_response.content,
                file_name=f"best_model_{task_str.lower()}.joblib",
                mime="application/octet-stream"
            )
        else:
            st.warning("Model file could not be fetched from the backend. The API might not be configured to return it yet.")
    except Exception as e:
        st.error("Failed to fetch model for downloading.")
