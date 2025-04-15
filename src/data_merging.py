"""
Data merging script for the telco churn prediction project.

This script loads the raw Excel files, merges them, and saves
the merged dataset as a CSV file.
"""

import logging
from pathlib import Path
import pandas as pd

# Logger configuration
import logger

# Get a logger
logger = logging.getLogger(__name__)

# Set paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not directory_path.exists():
        directory_path.mkdir(parents=True)
        logger.info(f"Created directory: {directory_path}")
    else:
        logger.info(f"Directory already exists: {directory_path}")

def clean_column_names(df):
    """
    Standardize columns names by removing leading/trailing spaces, converting to lowercase and replacing all non-alphanumeric characters with underscore
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", '_', regex=True)
        .str.replace(r"^_|_$", '', regex=True)
    )
    return df

def load_datasets():
    """Load raw Excel datasets into a dictionary"""
    logger.info("Loading datasets...")
    
    raw_files = {
        'demographics': RAW_DATA_DIR / "Telco_customer_churn_demographics.xlsx",
        'location': RAW_DATA_DIR / "Telco_customer_churn_location.xlsx",
        'population': RAW_DATA_DIR / "Telco_customer_churn_population.xlsx",
        'services': RAW_DATA_DIR / "Telco_customer_churn_services.xlsx",
        'status': RAW_DATA_DIR / "Telco_customer_churn_status.xlsx",
        'churn1': RAW_DATA_DIR / "CustomerChurn.xlsx",
        'churn2': RAW_DATA_DIR / "Telco_customer_churn.xlsx"
    }
    datasets = {}

    for name, file_path in raw_files.items():
        try:
            datasets[name] = pd.read_excel(file_path)
            logger.info(f"Loaded {name} dataset with shape {datasets[name].shape}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    # Standardize column names
    for name in datasets:
        datasets[name] = clean_column_names(datasets[name])
    
    return datasets

def check_primary_keys(datasets):
    """Check if customer_id can be used as primary key for merging datasets"""
    logger.info("Checking primary keys...")
    
    primary_datasets = ['demographics', 'location', 'services', 'status']
    
    s = set(datasets[primary_datasets[0]]['customer_id'])
    logger.info(f"Number of unique customer ids in {primary_datasets[0]}: {len(s)}")
    
    for name in primary_datasets[1:]:
        if s == set(datasets[name]['customer_id']):
            logger.info(f"{name} has the same customer ids")
        else:
            logger.warning(f"Warning! {name} does not have the same customer id set")
            return False
    
    return True

def merge_without_duplicates(df1, df2, df2_name):
    """Merge df1 and df2 while identifying and dropping columns with identical name and values"""
    common_cols = set(df1.columns) & set(df2.columns)

    # Identifying identical columns depends on the index sorting
    cols_to_drop = [col for col in common_cols if df1[col].equals(df2[col])]

    # Drop identical columns and merge
    df2 = df2.copy().drop(cols_to_drop, axis=1)
    df1 = df1.merge(df2, on='customer_id', how='outer')

    logger.info(f"Merging with {df2_name}")
    logger.info(f"Columns dropped when merging: {cols_to_drop}")
    logger.info(f"Resulting shape: {df1.shape}")

    return df1

def merge_datasets(datasets):
    """Merge primary datasets and population dataset"""
    logger.info("\nMerging datasets...")
    
    primary_datasets = ['demographics', 'location', 'services', 'status']
    
    # Set customer_id as index for primary datasets
    for name in primary_datasets:
        datasets[name].set_index('customer_id', inplace=True)
        datasets[name].sort_index(inplace=True)
    
    # Merge primary datasets
    merged_df = datasets[primary_datasets[0]].copy()
    logger.info(f"Starting with {primary_datasets[0]}")
    
    for name in primary_datasets[1:]:
        merged_df = merge_without_duplicates(merged_df, datasets[name], name)
    
    # Reset index to make customer_id a regular column
    merged_df = merged_df.reset_index()
    
    # Merge with population dataset
    merged_df = pd.merge(merged_df, datasets['population'].copy(), on='zip_code', how='left')
    
    # Set customer_id back as index
    merged_df.set_index('customer_id', inplace=True)
    
    return merged_df

def compare_with_combined_datasets(merged_df, datasets):
    """Compare the merged dataset with the two pre-consolidated datasets"""
    logger.info("Comparing with pre-consolidated datasets...")
    
    merged_df_cols = set(merged_df.columns)
    churn1_cols = set(datasets['churn1'].columns)
    churn2_cols = set(datasets['churn2'].columns)
    
    return {
        'in_churn1_not_in_merged': sorted(churn1_cols - merged_df_cols),
        'in_churn2_not_in_merged': sorted(churn2_cols - merged_df_cols),
        'in_merged_not_in_others': sorted(merged_df_cols - (churn1_cols | churn2_cols))
    }

def run_data_merging():
    """
    Main function to load, process, and save the merged dataset.
    Returns True if successful, False otherwise.
    """
    try:
        # Ensure the processed data directory exists
        create_directory(PROCESSED_DATA_DIR)
        
        # Load datasets
        datasets = load_datasets()
        
        # Check primary keys
        if not check_primary_keys(datasets):
            logger.error("Error: Primary keys do not match across datasets!")
            return False
        
        # Merge datasets
        merged_df = merge_datasets(datasets)
        
        # Check for duplicates
        if merged_df.duplicated().sum() > 0:
            logger.warning(f"Warning: {merged_df.duplicated().sum()} duplicated rows found in the merged dataset!")
            return False
        else:
            logger.info("No duplicated rows found in the merged dataset.")
        
        # Compare with combined datasets
        comparison = compare_with_combined_datasets(merged_df, datasets)
        logger.info(f"Columns in churn1 but not in merged_df: {comparison['in_churn1_not_in_merged']}")
        logger.info(f"Columns in churn2 but not in merged_df: {comparison['in_churn2_not_in_merged']}")
        logger.info(f"Columns in merged_df but not in the other two: {comparison['in_merged_not_in_others']}")
        
        # Save merged dataset
        output_path = PROCESSED_DATA_DIR / 'telco_merged.csv'
        merged_df.to_csv(output_path)
        logger.info(f"\nMerged dataset saved to {output_path}")
        logger.info(f"Shape: {merged_df.shape}")
        return True

    except Exception as e:
        logger.exception(f"Error during data merging: {str(e)}")
        return False

# For backward compatibility
def main():
    return run_data_merging()

if __name__ == "__main__":
    success = main()
    if not success:
        logger.warning("\nData merging script completed with errors. Review output and check notebooks.")
    else:
        logger.info("\nData merging script completed successfully.")