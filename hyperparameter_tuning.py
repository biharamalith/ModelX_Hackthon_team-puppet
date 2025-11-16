import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import config
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load preprocessed data for hyperparameter tuning."""
    preprocessed_file = 'modelx_preprocessed.csv'
    print(f"Loading preprocessed data from {preprocessed_file}...")
    try:
        df = pd.read_csv(preprocessed_file)
    except FileNotFoundError:
        print(f"ERROR: Preprocessed file not found: {preprocessed_file}")
        print("Please run the preprocessing script first.")
        return None, None, None
    
    print(f"Loaded preprocessed data: {df.shape}")
    print(f"Target distribution:\n{df[config.TARGET_COL].value_counts()}")
    
    # Separate features and target (data is already cleaned and imputed)
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]
    
    # Scale features
    scaler = StandardScaler()
    numeric_cols = [col for col in config.NUMERIC_FEATURES if col in X.columns]
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y, scaler

def tune_random_forest(X, y):
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    print("\n" + "="*60)
    print("Random Forest Hyperparameter Tuning (Fast Mode)")
    print("="*60)
    
    # Reduced parameter distributions for faster execution
    param_distributions = {
        'n_estimators': [200, 300, 500],  # Reduced from 4 to 3 options
        'max_depth': [20, 30, None],  # Reduced from 5 to 3 options
        'min_samples_split': [2, 5],  # Reduced from 3 to 2 options
        'min_samples_leaf': [1, 2],  # Reduced from 3 to 2 options
        'max_features': ['sqrt', 0.3],  # Reduced from 4 to 2 options
        'bootstrap': [True],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Base model
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # RandomizedSearchCV with reduced iterations and folds
    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=20,  # Reduced from 50 to 20 for faster execution
        scoring='roc_auc',
        cv=3,  # Reduced from 5 to 3 for faster execution
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nStarting hyperparameter search...")
    print("Testing 20 parameter combinations with 3-fold CV")
    print("Estimated time: 2-3 minutes\n")
    
    random_search.fit(X, y)
    
    print("\n" + "="*60)
    print("Tuning Complete!")
    print("="*60)
    print(f"Best ROC-AUC Score (CV): {random_search.best_score_:.4f}")
    print(f"\nBest Parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return random_search.best_estimator_, random_search

def evaluate_tuned_model(model, X, y):
    """Evaluate the tuned model with cross-validation."""
    print("\n" + "="*60)
    print("Evaluating Tuned Random Forest Model")
    print("="*60)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    
    print(f"\nCross-Validation ROC-AUC Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return cv_scores

def plot_tuning_results(random_search):
    """Plot the results of hyperparameter tuning."""
    results = pd.DataFrame(random_search.cv_results_)
    
    # Plot top 20 parameter combinations
    top_results = results.nlargest(20, 'mean_test_score')
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_results)), top_results['mean_test_score'], color='skyblue')
    plt.xlabel('Mean CV ROC-AUC Score')
    plt.ylabel('Parameter Combination Rank')
    plt.title('Top 20 Hyperparameter Combinations')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = config.REPORTS_DIR / 'rf_tuning_results.png'
    plt.savefig(plot_path)
    print(f"\nSaved tuning results plot to {plot_path}")
    plt.close()

def plot_feature_importance(model, feature_names):
    """Plot feature importance from tuned model."""
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Top 15 features
    top_features = feature_imp.head(15).sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(top_features['Feature'], top_features['Importance'], color='forestgreen')
    plt.title('Top 15 Feature Importances - Tuned Random Forest')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    plot_path = config.REPORTS_DIR / 'tuned_rf_feature_importance.png'
    plt.savefig(plot_path)
    print(f"Saved feature importance plot to {plot_path}")
    plt.close()
    
    # Save feature importance to CSV
    csv_path = config.REPORTS_DIR / 'feature_importance_tuned_rf.csv'
    feature_imp.to_csv(csv_path, index=False)
    print(f"Saved feature importance data to {csv_path}")
    
    return feature_imp

def compare_baseline_vs_tuned():
    """Compare baseline and tuned model performance."""
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    
    # Load comparison results
    baseline_results = pd.read_csv(config.MODEL_METRICS_FILE)
    rf_baseline = baseline_results[baseline_results['Model'] == 'Random Forest'].iloc[0]
    
    print("\nBaseline Random Forest:")
    print(f"  ROC-AUC: {rf_baseline['ROC-AUC']:.4f}")
    print(f"  CV Mean: {rf_baseline['CV Mean']:.4f} (+/- {rf_baseline['CV Std']:.4f})")
    
    return rf_baseline

def main():
    print("="*60)
    print("Random Forest Hyperparameter Tuning Pipeline")
    print("="*60)
    
    # Load data
    X, y, scaler = load_and_prepare_data()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Compare with baseline
    baseline_rf = compare_baseline_vs_tuned()
    
    # Tune Random Forest
    tuned_model, random_search = tune_random_forest(X, y)
    
    # Evaluate tuned model
    cv_scores = evaluate_tuned_model(tuned_model, X, y)
    
    # Plot results
    plot_tuning_results(random_search)
    feature_imp = plot_feature_importance(tuned_model, X.columns.tolist())
    
    # Save tuned model
    tuned_model_path = config.MODELS_DIR / 'tuned_random_forest.joblib'
    joblib.dump(tuned_model, tuned_model_path)
    print(f"\nTuned model saved to {tuned_model_path}")
    
    # Save scaler
    scaler_path = config.MODELS_DIR / 'scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Final comparison
    improvement = cv_scores.mean() - baseline_rf['CV Mean']
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    print(f"Baseline RF CV ROC-AUC: {baseline_rf['CV Mean']:.4f}")
    print(f"Tuned RF CV ROC-AUC:    {cv_scores.mean():.4f}")
    print(f"Improvement:            {improvement:+.4f} ({improvement/baseline_rf['CV Mean']*100:+.2f}%)")
    
    print("\nTop 10 Most Important Features:")
    print(feature_imp.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
