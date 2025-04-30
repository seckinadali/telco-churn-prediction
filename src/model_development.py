"""
Model development script for the telco churn prediction project.

This script implements the final modeling approach following the approach developed in the notebook '03_modeling_and_evaluation.ipynb'.

It involves the following steps:
1. Data preparation and preprocessing
2. Training and tuning XGBoost and Linear SVC models
3. Threshold adjustment for XGBoost to optimize for different business objectives
4. Final model evaluation on test set
5. Saving trained models and thresholds
"""

import sys
import logging
import pickle
from pathlib import Path

# Add project root to Python path to allow importing logger
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    make_scorer
)

import xgboost as xgb

# Logger configuration
import logger

# Get a logger
logger = logging.getLogger(__name__)

# Set paths
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not directory_path.exists():
        directory_path.mkdir(parents=True)
        logger.info(f"Created directory: {directory_path}")
    else:
        logger.info(f"Directory already exists: {directory_path}")

def load_data():
    """Load the cleaned dataset"""
    logger.info("Loading cleaned dataset...")
    
    try:
        file_path = PROCESSED_DATA_DIR / 'telco_cleaned.csv'
        df = pd.read_csv(file_path, index_col=0)
        logger.debug(f"Loaded dataframe with shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise

def prepare_data(df, test_size=0.2, validation_size=0.2, random_state=42):
    """
    Split data into train, validation and test sets, and create preprocessor

    Returns: (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)
    """
    logger.info("Preparing data for modeling...")
    
    # Separate target variable
    X = df.drop(['churn_value'], axis=1)
    y = df['churn_value']
    
    # Identify numerical and categorical features
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    logger.debug(f"Categorical features ({len(cat_features)}): {cat_features}")
    logger.debug(f"Numerical features ({len(num_features)}): {num_features}")
    
    # First split: training+validation and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=validation_size, 
        random_state=random_state, stratify=y_train_val
    )
    
    logger.debug(f"Training set size: {X_train.shape[0]} samples")
    logger.debug(f"Validation set size: {X_val.shape[0]} samples")
    logger.debug(f"Test set size: {X_test.shape[0]} samples")
    
    # Class distributions
    logger.debug(f"Training set class distribution: {y_train.value_counts(normalize=True) * 100}")
    logger.debug(f"Validation set class distribution: {y_val.value_counts(normalize=True) * 100}")
    logger.debug(f"Test set class distribution: {y_test.value_counts(normalize=True) * 100}")
    
    # Create preprocessor
    num_tr = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_tr = Pipeline([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_tr, num_features),
        ('cat', cat_tr, cat_features)
    ])
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

def tune_xgboost(X_train, y_train, preprocessor, random_state=42):
    """
    Tune XGBoost model based on findings from notebook experiments

    Returns: Tuple of (best_model, best_params)
    """
    logger.info("Tuning XGBoost model...")
    
    # Calculate positive class weight
    pos_class_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    logger.debug(f"Positive class weight: {pos_class_weight}")
    
    # Parameter grid based on notebook findings
    xgb_params = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.005, 0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7, 10],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
        'model__reg_alpha': [0, 1],
        'model__reg_lambda': [0, 1],
        'model__scale_pos_weight': [1, pos_class_weight]
    }
    
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBClassifier(
            random_state=random_state,
            eval_metric='logloss',
            n_jobs=-1
        ))
    ])
    
    # F1 scorer
    f1_scorer = make_scorer(f1_score)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    
    # Grid search
    grid_search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=xgb_params,
        n_iter=100,
        scoring=f1_scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Best model evaluation on validation set
    best_model = grid_search.best_estimator_
    
    # Log best parameters
    logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
    
    return best_model, grid_search.best_params_

def tune_linear_svc(X_train, y_train, preprocessor, random_state=42):
    """
    Tune Linear SVC model based on findings from notebook experiments

    Returns: Tuple of (best_model, best_params)
    """
    logger.info("Tuning Linear SVC model...")
    
    # Parameter grid based on notebook findings
    svc_params = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2'],
        'model__tol': [1e-4, 1e-5],
        'model__class_weight': ['balanced', None],
        'model__max_iter': [2000, 4000, 6000]
    }
    
    svc_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearSVC(
            random_state=random_state,
            dual=False
        ))
    ])
    
    # F1 scorer
    f1_scorer = make_scorer(f1_score)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    
    # Grid search
    grid_search = RandomizedSearchCV(
        svc_pipeline,
        param_distributions=svc_params,
        n_iter=100,
        scoring=f1_scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Best model evaluation on validation set
    best_model = grid_search.best_estimator_
    
    # Log best parameters
    logger.info(f"Best Linear SVC parameters: {grid_search.best_params_}")
    
    return best_model, grid_search.best_params_

def adjust_thresholds(xgb_model, X_val, y_val, plot_check=False):
    """
    Find optimal thresholds for XGBoost based on different business objectives
    
    Returns: Dictionary with optimal thresholds
    """
    logger.info("Adjusting thresholds for XGBoost model...")
    
    # Get prediction probabilities on validation set
    y_val_prob = xgb_model.predict_proba(X_val)[:, 1]
    
    # Calculate precision-recall curve values
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
    
    # Calculate F1 scores for each threshold
    f1_scores = [
        2 * (p * r) / (p + r) if (p + r) > 0 else 0
        for p, r in zip(precisions[:-1], recalls[:-1])
    ]
    
    # Find threshold that maximizes F1 score (balanced approach)
    optimal_f1_threshold_idx = np.argmax(f1_scores)
    optimal_f1_threshold = thresholds[optimal_f1_threshold_idx]
    logger.info(f"Optimal F1 threshold: {optimal_f1_threshold:.4f} (F1: {f1_scores[optimal_f1_threshold_idx]:.4f})")
    
    # Find threshold for high recall (e.g., 0.85 or higher)
    min_recall = 0.85
    recall_threshold_idx = np.argmin(np.abs(recalls - min_recall))
    recall_threshold = thresholds[recall_threshold_idx]
    logger.info(f"High recall threshold: {recall_threshold:.4f} (Recall: {recalls[recall_threshold_idx]:.4f})")
    
    # Find threshold for high precision (e.g., 0.85 or higher)
    min_precision = 0.85
    precision_threshold_idx = np.argmin(np.abs(precisions - min_precision))
    precision_threshold = thresholds[precision_threshold_idx]
    logger.info(f"High precision threshold: {precision_threshold:.4f} (Precision: {precisions[precision_threshold_idx]:.4f})")

    default_threshold = 0.5
    
    if plot_check:
        # Plot metrics separately
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, precisions[:-1], label='Precision')
        plt.plot(thresholds, recalls[:-1], label='Recall')
        plt.plot(thresholds, f1_scores, label='F1 Score', color='tab:brown')

        plt.axvline(x=default_threshold, color='tab:olive',
                    linestyle='--', alpha=0.8,
                    label=f"Default threshold: {default_threshold:.3f}")

        plt.axvline(x=optimal_f1_threshold, color='tab:red',
                    linestyle='--', alpha=0.8,
                    label=f"Optimal F1 threshold: {optimal_f1_threshold:.3f}")

        plt.axvline(x=recall_threshold, color='tab:green',
                    linestyle='--', alpha=0.8,
                    label=f"Recall ≥ 0.85 threshold: {recall_threshold:.3f}")

        plt.axvline(x=precision_threshold, color='tab:purple',
                    linestyle='--', alpha=0.8,
                    label=f"Precision ≥ 0.85 threshold: {precision_threshold:.3f}")

        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold vs XGBoost Metrics on Validation Set')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        plt.savefig(FIGURES_DIR / 'threshold_adjustment.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'default': 0.5,
        'optimal_f1': optimal_f1_threshold,
        'high_recall': recall_threshold,
        'high_precision': precision_threshold
    }

def evaluate_on_test(xgb_model, svc_model, thresholds, X_test, y_test, plot_check=False):
    """
    Evaluate final models on the test set
    
    Returns: Dictionary with evaluation results
    """
    logger.info("Evaluating models on test set...")
    
    results = {}
    
    # XGBoost evaluation
    y_test_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # Evaluate for each threshold
    for name, threshold in thresholds.items():
        y_test_pred_xgb = (y_test_prob_xgb >= threshold).astype(int)
        
        results[f'XGBoost ({name})'] = {
            'F1 Score': f1_score(y_test, y_test_pred_xgb),
            'Precision': precision_score(y_test, y_test_pred_xgb),
            'Recall': recall_score(y_test, y_test_pred_xgb),
            'Accuracy': accuracy_score(y_test, y_test_pred_xgb)
        }
        
        # Log results
        logger.info(f"XGBoost ({name}) performance:")
        logger.info(f"  F1 Score: {results[f'XGBoost ({name})']['F1 Score']:.4f}")
        logger.info(f"  Precision: {results[f'XGBoost ({name})']['Precision']:.4f}")
        logger.info(f"  Recall: {results[f'XGBoost ({name})']['Recall']:.4f}")
        logger.info(f"  Accuracy: {results[f'XGBoost ({name})']['Accuracy']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred_xgb)
        logger.debug(f"Confusion matrix for XGBoost ({name}):")
        logger.debug(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        logger.debug(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    # Linear SVC evaluation
    y_test_pred_svc = svc_model.predict(X_test)
    
    results['Linear SVC'] = {
        'F1 Score': f1_score(y_test, y_test_pred_svc),
        'Precision': precision_score(y_test, y_test_pred_svc),
        'Recall': recall_score(y_test, y_test_pred_svc),
        'Accuracy': accuracy_score(y_test, y_test_pred_svc)
    }
    
    # Log results
    logger.info(f"Linear SVC performance:")
    logger.info(f"  F1 Score: {results['Linear SVC']['F1 Score']:.4f}")
    logger.info(f"  Precision: {results['Linear SVC']['Precision']:.4f}")
    logger.info(f"  Recall: {results['Linear SVC']['Recall']:.4f}")
    logger.info(f"  Accuracy: {results['Linear SVC']['Accuracy']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred_svc)
    logger.debug(f"Confusion matrix for Linear SVC:")
    logger.debug(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    logger.debug(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    

    # Convert results to DataFrame for visualization
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'F1 Score': [results[k]['F1 Score'] for k in results],
        'Precision': [results[k]['Precision'] for k in results],
        'Recall': [results[k]['Recall'] for k in results],
        'Accuracy': [results[k]['Accuracy'] for k in results]
    })
    
    if plot_check:
        # Plot performance comparison
        plt.figure(figsize=(12, 6))
        results_df.set_index('Model').plot(kind='bar', figsize=(12, 6))
        plt.title('Model Performances on Test Set')
        plt.ylabel('Score')
        plt.xlabel('')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot Precision-Recall curves
        plt.figure(figsize=(12, 8))

        # Calculate PR curve for XGBoost
        precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_test_prob_xgb)
        pr_auc_xgb = auc(recall_xgb, precision_xgb)
        plt.plot(recall_xgb, precision_xgb, 
                label=f'XGBoost (AUC = {pr_auc_xgb:.3f})')

        # Get decision function scores for Linear SVC
        svc_decision_scores = svc_model.decision_function(X_test)

        # Calculate PR curve for Linear SVC
        precision_svc, recall_svc, _ = precision_recall_curve(y_test, svc_decision_scores)
        pr_auc_svc = auc(recall_svc, precision_svc)
        plt.plot(recall_svc, precision_svc, 
                label=f'Linear SVC (AUC = {pr_auc_svc:.3f})')

        # Mark different thresholds on the XGBoost PR curve:

        # Helper function to find points on the PR curve for specific thresholds
        def find_pr_threshold_point(thresh, probs):
            pred = (probs >= thresh).astype(int)
            prec = precision_score(y_test, pred)
            rec = recall_score(y_test, pred)
            return rec, prec

        # Default threshold (0.5)
        default_recall, default_precision = find_pr_threshold_point(0.5, y_test_prob_xgb)
        plt.scatter(default_recall, default_precision,
                    color='tab:olive', s=100, alpha=0.8,
                    label=f"Default threshold: {thresholds['default']:.3f}")

        # Optimal F1 threshold
        f1_recall, f1_precision = find_pr_threshold_point(thresholds['optimal_f1'], y_test_prob_xgb)
        plt.scatter(f1_recall, f1_precision,
                    color='tab:red', s=100, alpha=0.8,
                    label=f"Optimal F1 threshold: {thresholds['optimal_f1']:.3f}")

        # Recall-focused threshold
        recall_recall, recall_precision = find_pr_threshold_point(thresholds['high_recall'], y_test_prob_xgb)
        plt.scatter(recall_recall, recall_precision,
                    color='tab:green', s=100, alpha=0.8,
                    label=f"Recall ≥ 0.85 threshold: {thresholds['high_recall']:.3f}")

        # Precision-focused threshold
        precision_recall, precision_precision = find_pr_threshold_point(thresholds['high_precision'], y_test_prob_xgb)
        plt.scatter(precision_recall, precision_precision,
                    color='tab:purple', s=100, alpha=0.8,
                    label=f"Precision ≥ 0.85 threshold: {thresholds['high_precision']:.3f}")

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve with Different Threshold Strategies')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        plt.savefig(FIGURES_DIR / 'pr_curves_with_thresholds.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return True

def save_models(xgb_model, svc_model, thresholds):
    """Save trained models and thresholds to disk"""
    logger.info("Saving models to disk...")
    
    # Save XGBoost model
    with open(MODELS_DIR / 'xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    # Save Linear SVC model
    with open(MODELS_DIR / 'linearsvc_model.pkl', 'wb') as f:
        pickle.dump(svc_model, f)
    
    # Save thresholds
    with open(MODELS_DIR / 'thresholds.pkl', 'wb') as f:
        pickle.dump(thresholds, f)
    
    logger.info(f"Models and thresholds saved to {MODELS_DIR}")
    
    return True

def run_model_development(plot_check=False):
    """
    Main function to develop models.
    Returns True if successful, False otherwise.
    """
    try:
        # Ensure figures and models directories exist
        create_directory(MODELS_DIR)
        create_directory(FIGURES_DIR)
        
        # Load data
        df = load_data()
        
        # Prepare data for modeling
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = prepare_data(df)
        
        # Train and tune XGBoost model
        xgb_model, xgb_params = tune_xgboost(X_train, y_train, preprocessor)
        
        # Train and tune Linear SVC model
        svc_model, svc_params = tune_linear_svc(X_train, y_train, preprocessor)
        
        # Adjust thresholds for XGBoost
        thresholds = adjust_thresholds(xgb_model, X_val, y_val, plot_check=plot_check)
        
        # Fit models on complete training data (train + validation)
        X_train_full = pd.concat([X_train, X_val])
        y_train_full = pd.concat([y_train, y_val])
        
        # Extract best params
        xgb_best_params = {k.replace('model__', ''): v for k, v in xgb_params.items() if k.startswith('model__')}
        svc_best_params = {k.replace('model__', ''): v for k, v in svc_params.items() if k.startswith('model__')}
        
        # Create and fit final models
        final_xgb = Pipeline([
            ('preprocessor', preprocessor),
            ('model', xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1,
                **xgb_best_params
            ))
        ])
        
        final_svc = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearSVC(
                random_state=42,
                dual=False,
                **svc_best_params
            ))
        ])
        
        logger.info("Fitting final models on complete training data...")
        final_xgb.fit(X_train_full, y_train_full)
        final_svc.fit(X_train_full, y_train_full)
        
        # Evaluate models on test set
        evaluate_on_test(final_xgb, final_svc, thresholds, X_test, y_test, plot_check=plot_check)
        
        # Save models
        save_models(final_xgb, final_svc, thresholds)
        
        logger.info("Model development completed successfully!")
        return True
    
    except Exception as e:
        logger.exception(f"Error during model development: {str(e)}")
        return False

def main():
    return run_model_development()

if __name__ == "__main__":
    success = main()
    if not success:
        logger.warning("Model development script completed with errors. Review output.")
    else:
        logger.info("Model development script completed successfully.")