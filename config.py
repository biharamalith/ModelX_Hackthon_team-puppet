# --- config.py ---
# Central configuration file for the ModelX project.

import numpy as np

# --- 1. Project Settings ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5  # Number of folds for StratifiedKFold cross-validation

# --- 2. File Paths ---
# Use pathlib for robust path handling
from pathlib import Path
DATA_DIR = Path('data')
REPORTS_DIR = Path('reports')
MODELS_DIR = Path('models')

INPUT_FILE = DATA_DIR / 'Dementia Prediction Dataset.csv'
EDA_REPORT_FILE = REPORTS_DIR / 'modelx_eda_report.html'
BEST_MODEL_FILE = MODELS_DIR / 'best_model.joblib'
MODEL_METRICS_FILE = REPORTS_DIR / 'model_comparison_results.csv'

# --- 3. Target & Feature Definitions ---
TARGET_COL = 'NACCUDSD'

# Whitelist features based on hackathon rules
WHITELIST_FEATURES = [
    # Form A1: Demographics
    'NACCAGE', 'NACCAGEB', 'SEX', 'EDUC', 'MARISTAT', 'NACCLIVS', 'RACE', 'HISPANIC', 'HANDED',
    
    # Form A3: Family History
    'NACCFAM', 'NACCMOM', 'NACCDAD',
    
    # Form A5: Self-Reported Health History
    'TOBAC100', 'SMOKYRS', 'ALCOHOL', 'CVHATT', 'CVAFIB', 'CVCHF', 'CBSTROKE',
    'CBTIA', 'DIABETES', 'HYPERTEN', 'HYPERCHO', 'NACCTBI', 'APNEA', 'DEP2YRS',
    
    # Form B1: Simple Physical Measures
    'NACCBMI', 'HEIGHT', 'WEIGHT', 'HEARING', 'HEARAID', 'BPSYS', 'BPDIAS',
]

# Define feature types for preprocessing pipelines
NUMERIC_FEATURES = [
    'NACCAGE', 'NACCAGEB', 'EDUC', 'SMOKYRS', 'NACCBMI', 'HEIGHT', 
    'WEIGHT', 'BPSYS', 'BPDIAS'
]

CATEGORICAL_FEATURES = [
    'SEX', 'MARISTAT', 'NACCLIVS', 'RACE', 'HISPANIC', 'HANDED', 'NACCFAM', 
    'NACCMOM', 'NACCDAD', 'TOBAC100', 'ALCOHOL', 'CVHATT', 'CVAFIB', 'CVCHF', 
    'CBSTROKE', 'CBTIA', 'DIABETES', 'HYPERTEN', 'HYPERCHO', 'NACCTBI', 
    'APNEA', 'DEP2YRS', 'HEARING', 'HEARAID'
]

# --- 4. Preprocessing Settings ---
# NACC-specific missing value codes
MISSING_CODES = [
    -4, 8, 9, 88, 99, 888, 999, 8888, 9999
]

# Target variable mapping
# 0 = 'Not at Risk' (1: Normal cognition)
# 1 = 'At Risk' (2: Impaired-not-MCI, 3: MCI, 4: Dementia)
TARGET_MAP = {1: 0, 2: 1, 3: 1, 4: 1}

# Ensure all directories exist
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)