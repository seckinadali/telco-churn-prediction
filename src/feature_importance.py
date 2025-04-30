"""
Feature importance analysis script for the telco churn prediction project.

This script loads the trained models, extracts and visualizes feature importance
using multiple techniques:
- XGBoost built-in importance
- SHAP values
- Linear SVC coefficients

It follows the approach developed in the '03_modeling_and_evaluation.ipynb' notebook.
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

import shap

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

def load_models():
    """Load the trained XGBoost and Linear SVC models"""
    logger.info("Loading trained models...")
    
    try:
        # Load XGBoost model
        with open(MODELS_DIR / 'xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        
        # Load Linear SVC model
        with open(MODELS_DIR / 'linearsvc_model.pkl', 'rb') as f:
            svc_model = pickle.load(f)
        
        logger.info("Models loaded successfully")
        return xgb_model, svc_model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

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

def get_feature_names(df, preprocessor):
    """Extract feature names after preprocessing"""
    logger.info("Extracting feature names...")
    
    # Get categorical and numerical features
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target if present
    if 'churn_value' in num_features:
        num_features.remove('churn_value')

    # Create mapping for one-hot encoded features
    encoded_features = []
    
    # Get encoded feature names
    for feature in cat_features:
        # Get the categories from the encoder
        try:
            categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[cat_features.index(feature)]
            # Skip the first since we applied drop='first' in OneHotEncoder
            for category in categories[1:]:
                encoded_name = f"{feature}_{category}"
                encoded_features.append(encoded_name)
        except (KeyError, IndexError) as e:
            logger.warning(f"Error getting categories for {feature}: {e}")
            continue
    
    # Combine all feature names
    all_features = num_features + encoded_features
    
    logger.debug(f"Extracted {len(all_features)} feature names after preprocessing")
    return all_features, num_features, cat_features

def analyze_xgboost_importance(xgb_model, all_features, cat_features, num_features, plot_check=True):
    """
    Analyze and visualize XGBoost built-in feature importance
    
    Returns: Tuple of (top_individual_features_df, top_aggregated_features_df)
    """
    logger.info("Analyzing XGBoost feature importance...")
    
    # Extract the XGBoost model from the pipeline
    model = xgb_model.named_steps['model']
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a dataframe for individual feature importances
    imp_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    })
    
    imp_df = imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Get top 10 individual features
    top_individual_features = imp_df.head(10).copy()
    
    # Create a dataframe for aggregated categorical feature importances
    agg_rows = []
    
    # Add numerical features as they are
    for feature in num_features:
        idx = all_features.index(feature)
        agg_rows.append({
            'Feature': feature,
            'Importance': importances[idx],
            'Type': 'Numerical'
        })
    
    # Aggregate categorical features
    for feature in cat_features:
        # Sum importance of all one-hot encoded features from this category
        total_importance = sum(imp_df[imp_df['Feature'].apply(
            lambda x: x.startswith(feature + '_'))]['Importance'])
        
        agg_rows.append({
            'Feature': feature,
            'Importance': total_importance,
            'Type': 'Categorical'
        })
    
    agg_imp_df = pd.DataFrame(agg_rows)
    agg_imp_df = agg_imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Get top 10 aggregated features
    top_aggregated_features = agg_imp_df.head(10).copy()
    
    if plot_check:
        # Plot feature importances
        n_features = min(20, len(imp_df))
        fig, axes = plt.subplots(2, 1, figsize=(12, 16))
        
        # Individual features
        sns.barplot(data=imp_df.head(n_features), y='Feature', x='Importance', color='steelblue', ax=axes[0])
        axes[0].set_xlabel('')
        axes[0].set_ylabel('Individual Feature')
        axes[0].set_title(f'Top {n_features} Individual Feature Importances')
        axes[0].grid(axis='x', linestyle='--', alpha=0.7)
        
        # Aggregated features
        sns.barplot(data=agg_imp_df.head(n_features), y='Feature', x='Importance', color='steelblue', ax=axes[1])
        axes[1].set_xlabel('Importance')
        axes[1].set_ylabel('Aggregated Feature')
        axes[1].set_title(f'Top {n_features} Aggregated Feature Importances')
        axes[1].grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.suptitle(f"XGBoost Feature Importances", fontsize=16, y=1)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'xgboost_feature_importances.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"XGBoost feature importance analysis saved to {FIGURES_DIR}")
    
    # Return the top features dataframes
    return top_individual_features, top_aggregated_features

def analyze_shap_values(xgb_model, df, all_features):
    """Analyze and visualize SHAP values for XGBoost"""
    logger.info("Analyzing SHAP values...")
    
    try:
        # Extract the preprocessor and model from the pipeline
        preprocessor = xgb_model.named_steps['preprocessor']
        model = xgb_model.named_steps['model']
        
        # Prepare data for SHAP analysis
        X = df.drop(['churn_value'], axis=1)
        y = df['churn_value']
        
        # Use test data for SHAP analysis
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Get a stratified sample for SHAP analysis to keep computation manageable
        from sklearn.model_selection import StratifiedKFold
        
        def get_stratified_sample(X, y, sample_size=1000):
            """Get a stratified sample of data for SHAP"""
            # If dataset is smaller than sample_size, return everything
            if len(X) <= sample_size:
                return preprocessor.transform(X)
            
            stratify_split = StratifiedKFold(
                # Ensure fold size is close to sample_size
                n_splits=max(2, int(len(X) / sample_size)),
                shuffle=True, random_state=42
            )
            
            # Get indices for the first fold
            train_idx, test_idx = next(stratify_split.split(X, y))
            
            return preprocessor.transform(X.iloc[test_idx])
        
        # Get sample data
        X_sample = get_stratified_sample(X_test, y_test)
        
        # Create SHAP explainer
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)
        
        # Plot SHAP summary
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=all_features, show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        return True
    
    except Exception as e:
        logger.error(f"Error in SHAP analysis: {e}")
        return None

def analyze_linear_svc_coefficients(svc_model, all_features, plot_check=True):
    """
    Analyze and visualize Linear SVC coefficients
    
    Returns: DataFrame with top 10 features by coefficient magnitude
    """
    logger.info("Analyzing Linear SVC coefficients...")
    
    # Extract the SVC model from the pipeline
    model = svc_model.named_steps['model']
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Create df for coefficient values
    coef_df = pd.DataFrame({
        'Feature': all_features,
        'Coefficient': coefficients
    })
    
    # Sort by absolute value of coefficients
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).reset_index(drop=True)
    
    # Get top 10 features by absolute coefficient value
    top_svc_features = coef_df.head(10).copy()
    
    if plot_check:
        # Plot top n coefficients
        n_features = min(20, len(coef_df))
        plt.figure(figsize=(12, 10))
        
        # Create color palette based on coefficient sign
        colors = ['tab:red' if x < 0 else 'tab:blue' for x in coef_df.head(n_features)['Coefficient']]
        
        sns.barplot(
            data=coef_df.head(n_features), 
            y='Feature', 
            x='Coefficient',
            palette=colors
        )
        plt.title(f'Top {n_features} Feature Coefficients - Linear SVC')
        plt.axvline(x=0, color='gray', linestyle='--')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'linear_svc_coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Linear SVC coefficient analysis saved to {FIGURES_DIR}")
    
    # Return the top features dataframe
    return top_svc_features

def run_feature_importance(plot_check=True, shap_check=True):
    """
    Main function to analyze feature importance.
    Returns True if successful, False otherwise.
    """
    try:
        # Ensure figures directory exists
        create_directory(FIGURES_DIR)
        
        # Load models and data
        xgb_model, svc_model = load_models()
        df = load_data()
        
        # Get preprocessor from XGBoost model
        preprocessor = xgb_model.named_steps['preprocessor']
        
        # Get feature names after preprocessing
        all_features, num_features, cat_features = get_feature_names(df, preprocessor)
        
        # Analyze XGBoost feature importance
        top_individual_features, top_aggregated_features = analyze_xgboost_importance(
            xgb_model, all_features, cat_features, num_features, plot_check=plot_check
        )
        
        # Print top 10 features to console
        logger.info("TOP 10 XGBOOST INDIVIDUAL FEATURES")
        for i, row in top_individual_features.iterrows():
            logger.info(f"{i+1}. {row['Feature']}: {row['Importance']:.6f}")
        
        logger.info("TOP 10 XGBOOST AGGREGATED FEATURES")
        for i, row in top_aggregated_features.iterrows():
            feature_type = row['Type']
            logger.info(f"{i+1}. {row['Feature']} ({feature_type}): {row['Importance']:.6f}")

        # Analyze SHAP values (resource intensive)
        if shap_check:
            analyze_shap_values(xgb_model, df, all_features)
        
        # Analyze Linear SVC coefficients
        top_svc_features = analyze_linear_svc_coefficients(
            svc_model, all_features, plot_check=plot_check
        )
        
        # Print top 10 SVC features to console
        logger.info("TOP 10 LINEAR SVC FEATURES BY COEFFICIENT MAGNITUDE")
        for i, row in top_svc_features.iterrows():
            direction = "Decreases" if row['Coefficient'] < 0 else "Increases"
            logger.info(f"{i+1}. {row['Feature']}: {row['Coefficient']:.6f} ({direction} churn probability)")
        
        logger.info("Feature importance analysis completed successfully!")
        return True
    
    except Exception as e:
        logger.exception(f"Error during feature importance analysis: {str(e)}")
        return False

def main():
    return run_feature_importance()

if __name__ == "__main__":
    success = main()
    if not success:
        logger.warning("Feature importance analysis script completed with errors. Review output.")
    else:
        logger.info("Feature importance analysis script completed successfully.")