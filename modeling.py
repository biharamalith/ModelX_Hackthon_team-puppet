import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib
import config
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

def load_and_prepare_data():
    """Load preprocessed data for modeling."""
    preprocessed_file = 'modelx_preprocessed.csv'
    print(f"Loading preprocessed data from {preprocessed_file}...")
    try:
        df = pd.read_csv(preprocessed_file)
    except FileNotFoundError:
        print(f"ERROR: Preprocessed data file not found at {preprocessed_file}")
        print("Please run the preprocessing script first.")
        return None, None, None, None
    
    print(f"Loaded preprocessed data: {df.shape}")
    print(f"Target distribution:\n{df[config.TARGET_COL].value_counts()}")
    
    # Separate features and target (data is already cleaned and imputed)
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = [col for col in config.NUMERIC_FEATURES if col in X_train.columns]
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train Logistic Regression, Random Forest, and XGBoost models."""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    
    results = []
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  CV ROC-AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # Classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'ROC-AUC': roc_auc,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        })
        
        # Track best model
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = (name, model)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, name)
        
        # Plot ROC curve
        plot_roc_curve(y_test, y_pred_proba, name)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.MODEL_METRICS_FILE, index=False)
    print(f"\n{'='*50}")
    print(f"Model comparison results saved to {config.MODEL_METRICS_FILE}")
    print(f"\nBest Model: {best_model[0]} with ROC-AUC: {best_score:.4f}")
    
    # Save best model
    joblib.dump(best_model[1], config.BEST_MODEL_FILE)
    print(f"Best model saved to {config.BEST_MODEL_FILE}")
    
    return results_df, best_model

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix for a model."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = model_name.replace(' ', '_').lower()
    plot_path = config.REPORTS_DIR / f'confusion_matrix_{filename}.png'
    plt.savefig(plot_path)
    print(f"  Saved confusion matrix to {plot_path}")
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Plot ROC curve for a model."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filename = model_name.replace(' ', '_').lower()
    plot_path = config.REPORTS_DIR / f'roc_curve_{filename}.png'
    plt.savefig(plot_path)
    print(f"  Saved ROC curve to {plot_path}")
    plt.close()

def main():
    print("="*50)
    print("ModelX - Model Training Pipeline")
    print("Models: Logistic Regression, Random Forest, XGBoost")
    print("="*50)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    if X_train is None:
        print("Failed to load data. Exiting...")
        return
    
    # Train and evaluate models
    results_df, best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*50)
    print("Model Training Complete!")
    print("="*50)
    print("\nFinal Results Summary:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
