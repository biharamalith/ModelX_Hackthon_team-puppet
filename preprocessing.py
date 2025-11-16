import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import config

def preprocess_data():
    """
    Preprocess the raw dataset: select features, handle missing codes, 
    binarize target, and impute missing values.
    """
    input_file = 'data/Dementia Prediction Dataset.csv'
    output_file = 'modelx_preprocessed.csv'
    
    print("="*60)
    print("Data Preprocessing Pipeline")
    print("="*60)
    
    # --- 1. Load the full dataset ---
    try:
        print(f"\nLoading full dataset from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Original shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: The file '{input_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    
    # --- 2. Select whitelist features + target ---
    columns_to_keep = config.WHITELIST_FEATURES + [config.TARGET_COL]
    available_columns = [col for col in columns_to_keep if col in df.columns]
    
    if config.TARGET_COL not in available_columns:
        print(f"ERROR: Target variable '{config.TARGET_COL}' not found in dataset.")
        return
    
    print(f"\nSelecting {len(available_columns)} columns (whitelist + target)...")
    df = df[available_columns]
    print(f"Selected shape: {df.shape}")
    
    # --- 3. Handle the Target Variable ---
    print(f"\nProcessing target variable '{config.TARGET_COL}'...")
    
    # Binarize: 0 = Not at Risk (1), 1 = At Risk (2, 3, 4)
    df[config.TARGET_COL] = df[config.TARGET_COL].map(config.TARGET_MAP)
    
    # Drop rows with missing target
    initial_rows = df.shape[0]
    df = df.dropna(subset=[config.TARGET_COL])
    dropped_rows = initial_rows - df.shape[0]
    print(f"Dropped {dropped_rows} rows with missing target variable.")
    
    # Convert to integer
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)
    
    print(f"\nTarget distribution:")
    print(df[config.TARGET_COL].value_counts())
    print(f"\nClass proportions:")
    print(df[config.TARGET_COL].value_counts(normalize=True))
    
    # --- 4. Replace special missing codes with NaN ---
    feature_cols = [col for col in df.columns if col != config.TARGET_COL]
    
    print(f"\nReplacing special missing codes {config.MISSING_CODES} with NaN...")
    for col in feature_cols:
        df[col] = df[col].replace(config.MISSING_CODES, np.nan)
    
    missing_before = df[feature_cols].isnull().sum().sum()
    print(f"Total missing values after replacement: {missing_before}")
    
    # --- 5. Feature Engineering: Calculate Missing BMI ---
    if 'NACCBMI' in df.columns and 'WEIGHT' in df.columns and 'HEIGHT' in df.columns:
        bmi_missing_mask = df['NACCBMI'].isnull()
        height_valid = (df['HEIGHT'].notnull()) & (df['HEIGHT'] > 0)
        weight_valid = df['WEIGHT'].notnull()
        fill_mask = bmi_missing_mask & height_valid & weight_valid
        
        if fill_mask.sum() > 0:
            print(f"\nCalculating {fill_mask.sum()} missing BMI values from HEIGHT and WEIGHT...")
            # BMI = (weight_kg / height_m^2) * 703 for lbs/inches
            df.loc[fill_mask, 'NACCBMI'] = (df.loc[fill_mask, 'WEIGHT'] / (df.loc[fill_mask, 'HEIGHT']**2)) * 703
    
    # --- 6. Imputation ---
    print("\nImputing missing values...")
    
    # Identify numeric and categorical features
    numerical_features = [col for col in config.NUMERIC_FEATURES if col in df.columns]
    categorical_features = [col for col in feature_cols if col not in numerical_features]
    
    # Create imputers
    median_imputer = SimpleImputer(strategy='median')
    mode_imputer = SimpleImputer(strategy='most_frequent')
    
    # Impute numerical features with median
    if numerical_features:
        df[numerical_features] = median_imputer.fit_transform(df[numerical_features])
        print(f"  ✓ Imputed {len(numerical_features)} numerical features with median")
    
    # Impute categorical features with mode
    if categorical_features:
        df[categorical_features] = mode_imputer.fit_transform(df[categorical_features])
        print(f"  ✓ Imputed {len(categorical_features)} categorical features with mode")
    
    # --- 7. Final Check ---
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    
    missing_after = df.isnull().sum().sum()
    if missing_after == 0:
        print("✓ No missing values remain")
    else:
        print(f"⚠ Warning: {missing_after} missing values still present")
    
    # --- 8. Save preprocessed data ---
    df.to_csv(output_file, index=False)
    print(f"\n✓ Preprocessed data saved to: {output_file}")
    print(f"  Final shape: {df.shape}")
    print(f"  Features: {df.shape[1] - 1}")
    print(f"  Samples: {df.shape[0]}")
    
    # Display summary
    print(f"\n{'='*60}")
    print("Data Summary:")
    print(f"{'='*60}")
    print(df.info())
    
    return df

if __name__ == "__main__":
    preprocess_data()
