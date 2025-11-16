"""Utilities for lazily downloading and loading large model artifacts."""

import gdown
import joblib
import streamlit as st
from pathlib import Path


MODEL_FILES = {
    'tuned_random_forest.joblib': {
        'id': '1Z3RMLFn934osIzz5zVYJoa4VbCdDIHQD',
        'min_bytes': 400_000_000  # ~530 MB after compression
    },
    'scaler.joblib': {
        'id': '1F56EmPfF3VhaP4M7V6bhzsi14Z7KiDka',
        'min_bytes': 100_000      # ~120 KB
    }
}

MODEL_DIR = Path('../models')

def download_from_gdrive(file_id, destination):
    """Download file from Google Drive using gdown"""
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, str(destination), quiet=False, fuzzy=True)


def is_file_valid(path: Path, min_bytes: int) -> bool:
    """Return True if file exists and matches the expected minimum size."""
    return path.exists() and path.stat().st_size >= min_bytes

def ensure_models_exist():
    """Download models if they don't exist"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, meta in MODEL_FILES.items():
        filepath = MODEL_DIR / filename
        if is_file_valid(filepath, meta['min_bytes']):
            continue

        if filepath.exists():
            filepath.unlink(missing_ok=True)

        st.info(f"📥 Downloading {filename}... This may take a few minutes on first run.")
        try:
            download_from_gdrive(meta['id'], filepath)
            if is_file_valid(filepath, meta['min_bytes']):
                size_mb = filepath.stat().st_size / (1024 * 1024)
                st.success(f"✅ {filename} downloaded successfully ({size_mb:.1f} MB)")
            else:
                st.error(f"❌ {filename} looks incomplete. Please retry later.")
                return False
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
