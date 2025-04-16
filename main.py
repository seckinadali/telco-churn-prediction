"""
Run the complete pipeline for the telco churn prediction project.

This script orchestrates the entire pipeline by running the following scripts in sequence:
- data_merging.py,
- data_cleaning.py,
- model_development.py,
- threshold_adjustment.py,
- test_evaluation.py,
- feature_imortance.py
"""

import sys
import logging

from src.data_merging import run_data_merging
from src.data_cleaning import run_data_cleaning

# Logger configuration
import logger

# Get a logger
logger = logging.getLogger(__name__)

# Global configuration
PLOT_CHECK = True  # Set to True if you want to generate plots

def main():
    """
    Main function that orchestrates the execution of project components.
    """
    logger.info("Starting Telco Churn Prediction pipeline...")
    
    # Step 1: Data Merging
    logger.info("\n=== STEP 1: DATA MERGING ===")
    data_merging_success = run_data_merging()
    
    if not data_merging_success:
        logger.error("Data merging failed. Stopping the pipeline.")
        return False
    
    # Step 2: Data Cleaning and EDA
    logger.info("\n=== STEP 2: DATA CLEANING AND EDA ===")
    data_cleaning_success = run_data_cleaning(plot_check=PLOT_CHECK)
    
    if not data_cleaning_success:
        logger.error("Data cleaning failed. Stopping the pipeline.")
        return False
    
    # TODO: Remaining steps
    
    logger.info("\nTelco Churn Prediction pipeline completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)
