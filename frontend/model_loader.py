import os
import gdown
import joblib
import streamlit as st
from pathlib import Path


MODEL_FILES = {
    'tuned_random_forest.joblib': '1Z3RMLFn934osIzz5zVYJoa4VbCdDIHQD',
    'scaler.joblib': '1F56EmPfF3VhaP4M7V6bhzsi14Z7KiDka'
}

MODEL_DIR = Path('../models')

def download_from_gdrive(file_id, destination):
    """Download file from Google Drive using gdown"""
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, str(destination), quiet=False)

def ensure_models_exist():
    """Download models if they don't exist"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, file_id in MODEL_FILES.items():
        filepath = MODEL_DIR / filename
        
        if not filepath.exists():
            st.info(f"📥 Downloading {filename}... This may take a few minutes on first run.")
            try:
                download_from_gdrive(file_id, filepath)
                st.success(f"✅ {filename} downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to download {filename}: {str(e)}")
                return False
    
    return True

@st.cache_resource
def load_model_artifacts():
    """Load model and scaler, downloading if necessary"""
    if not ensure_models_exist():
        st.error("⚠️ Failed to load models. Please contact support.")
        return None, None
    
    try:
        model = joblib.load(MODEL_DIR / 'tuned_random_forest.joblib')
        scaler = joblib.load(MODEL_DIR / 'scaler.joblib')
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        return None, None
