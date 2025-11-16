import requests
import joblib
import streamlit as st
from pathlib import Path


MODEL_URLS = {
    'tuned_random_forest.joblib': 'https://drive.google.com/uc?export=download&id=1Z3RMLFn934osIzz5zVYJoa4VbCdDIHQD',
    'scaler.joblib': 'https://drive.google.com/uc?export=download&id=1F56EmPfF3VhaP4M7V6bhzsi14Z7KiDka'
}

MODEL_DIR = Path('../models')

def download_file(url, destination):
    """Download file from Google Drive with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    progress_bar = st.progress(0)
    downloaded = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress_bar.progress(downloaded / total_size)
    
    progress_bar.empty()

def ensure_models_exist():
    """Download models if they don't exist"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, url in MODEL_URLS.items():
        filepath = MODEL_DIR / filename
        
        if not filepath.exists():
            st.info(f"📥 Downloading {filename}... This may take a few minutes on first run.")
            try:
                download_file(url, filepath)
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
