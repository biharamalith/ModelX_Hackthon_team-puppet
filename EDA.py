import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import config  # Import our central configuration
import warnings

# --- 0. Setup ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

def load_and_clean_raw_data():
    """Loads raw data, selects whitelist, and handles known missing codes."""
    print(f"Loading raw data from {config.INPUT_FILE}...")
    try:
        df = pd.read_csv(config.INPUT_FILE)
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {config.INPUT_FILE}")
        print("Please place the 'modelx .xlsx - Sheet1.csv' file in a 'data/' directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
        
    # Select only the columns we are allowed to use
    cols_to_load = config.WHITELIST_FEATURES + [config.TARGET_COL]
    
    # Ensure all columns exist, dropping any from our list that don't
    available_cols = [col for col in cols_to_load if col in df.columns]
    missing_cols = [col for col in cols_to_load if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: The following columns were not found in the CSV: {missing_cols}")
        
    df = df[available_cols]
    
    # Replace all NACC-specific missing codes with standard np.nan
    print(f"Replacing special missing codes {config.MISSING_CODES}...")
    df.replace(config.MISSING_CODES, np.nan, inplace=True)
    
    # Binarize the target variable for EDA
    df[config.TARGET_COL] = df[config.TARGET_COL].map(config.TARGET_MAP)
    
    return df

def generate_eda_plots(df):
    """Generates and saves key visualizations for the report."""
    print("Generating EDA plots...")
    
    # 1. Target Distribution Plot
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(
        x=config.TARGET_COL, 
        data=df, 
        palette=['#4c72b0', '#c44e52']
    )
    plt.title('Target Variable Distribution (Class Imbalance)', fontsize=16)
    plt.xlabel('Dementia Risk', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=['0: Not at Risk', '1: At Risk'])
    
    # Add percentages
    total = len(df[config.TARGET_COL].dropna())
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=12, xytext=(0, 5), textcoords='offset points')
        
    plot_path = config.REPORTS_DIR / 'target_distribution.png'
    plt.savefig(plot_path)
    print(f"Saved target distribution plot to {plot_path}")
    plt.close()

    # 2. Correlation Heatmap
    # [Image of a data correlation heatmap]
    plt.figure(figsize=(16, 12))
    numeric_df = df[config.NUMERIC_FEATURES]
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Numeric Feature Correlation Heatmap', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plot_path = config.REPORTS_DIR / 'correlation_heatmap.png'
    plt.savefig(plot_path)
    print(f"Saved correlation heatmap to {plot_path}")
    plt.close()

    # 3. Missing Values Bar Chart
    plt.figure(figsize=(12, 8))
    missing_pct = df.isnull().sum() / len(df) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if not missing_pct.empty:
        sns.barplot(x=missing_pct.index, y=missing_pct.values, palette='viridis')
        plt.title('Percentage of Missing Values by Feature (Whitelist)', fontsize=16)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Percent Missing (%)', fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plot_path = config.REPORTS_DIR / 'missing_values.png'
        plt.savefig(plot_path)
        print(f"Saved missing values plot to {plot_path}")
        plt.close()
    else:
        print("No missing values found in whitelist columns.")

def generate_profiling_report(df):
    """Generates a comprehensive ydata-profiling report."""
    print(f"Generating profiling report... (This may take a few minutes)")
    profile = ProfileReport(
        df, 
        title="ModelX Dementia Risk - EDA Report",
        explorative=True
    )
    profile.to_file(config.EDA_REPORT_FILE)
    print(f"Successfully saved EDA report to {config.EDA_REPORT_FILE}")

def main():
    df = load_and_clean_raw_data()
    
    if df is not None:
        generate_eda_plots(df)
        generate_profiling_report(df)
        print("\nEDA script finished successfully.")

if __name__ == "__main__":
    main()