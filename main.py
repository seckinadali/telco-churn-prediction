"""
Run the complete pipeline for the telco churn prediction project.

This script orchestrates the entire pipeline by running the following scripts in sequence:
- data_merging.py
- data_cleaning.py
- model_development.py
- feature_imortance.py
"""

import sys
import logging

from src.data_merging import run_data_merging
from src.data_cleaning import run_data_cleaning
from src.model_development import run_model_development
from src.feature_importance import run_feature_importance

# Logger configuration
import logger

# Get a logger
logger = logging.getLogger(__name__)

# Global configuration
PLOT_CHECK = True
SHAP_CHECK = True

def main():
    """
    Main function that orchestrates the execution of project components.
    """
    logger.info("Starting Telco Churn Prediction pipeline...")
    
    # Step 1: Data Merging
    logger.info("=== STEP 1: DATA MERGING ===")
    data_merging_success = run_data_merging()
    
    if not data_merging_success:
        logger.error("Data merging failed. Stopping the pipeline.")
        return False
    
    # Step 2: Data Cleaning and EDA
    logger.info("=== STEP 2: DATA CLEANING AND EDA ===")
    data_cleaning_success = run_data_cleaning(plot_check=PLOT_CHECK)
    
    if not data_cleaning_success:
        logger.error("Data cleaning failed. Stopping the pipeline.")
        return False
    
    # Step 3: Model Development
    logger.info("=== STEP 3: MODEL DEVELOPMENT ===")
    model_development_success = run_model_development(plot_check=PLOT_CHECK)
    
    if not model_development_success:
        logger.error("Model development failed. Stopping the pipeline.")
        return False
    
    # Step 4: Feature Importance
    logger.info("=== STEP 4: FEATURE IMPORTANCE ===")
    feature_importance_success = run_feature_importance(plot_check=PLOT_CHECK, shap_check=SHAP_CHECK)
    
    if not feature_importance_success:
        logger.error("Feature importance failed. Stopping the pipeline.")
        return False
    
    logger.info("Telco Churn Prediction pipeline completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)
