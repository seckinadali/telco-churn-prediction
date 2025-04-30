"""
Data cleaning script for the telco churn prediction project.

This script loads the merged dataset, performs exploratory data analysis and feature engineering, and saves the cleaned dataset ready for modeling following the approach developed in the notebook '02_data_cleaning_and_eda.ipynb'.

The script follows these steps:
1. Replace missing values in categorical features with meaningful values
2. Remove redundant and duplicate features
3. Encode binary variables for consistency
4. Analyze correlations and address multicollinearity
5. Transform skewed numerical features
6. Check for and handle data leakage
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path to allow importing logger
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress matplotlib categorical warnings that occur when plotting numerical data 
# as categories. These warnings are informational only and don't indicate actual 
# problems with the visualization output.
logging.getLogger('matplotlib.category').setLevel(logging.WARNING)

# Logger configuration
import logger

# Get a logger
logger = logging.getLogger(__name__)

# Set paths
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not directory_path.exists():
        directory_path.mkdir(parents=True)
        logger.info(f"Created directory: {directory_path}")
    else:
        logger.info(f"Directory already exists: {directory_path}")

def load_data():
    """Load the merged dataset"""
    logger.info("Loading merged dataset...")
    
    try:
        file_path = PROCESSED_DATA_DIR / 'telco_merged.csv'
        df = pd.read_csv(file_path, index_col=0)
        logger.debug(f"Loaded dataframe with shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise

def missing_values(df):
    """Generate a dataframe containing info on missing values"""
    missing = df.isna().sum()
    missing_pct = round(100 * (missing / len(df)), 2)

    res = pd.DataFrame({
        'missing_count': missing.values,
        'missing_pct': missing_pct,
    })

    return res[res['missing_count'] > 0]

def handle_missing_values(df):
    """Handle missing values in the DataFrame"""
    logger.info("Handling missing values...")
    
    # Check missing values
    missing_report = missing_values(df)
    if not missing_report.empty:
        logger.info("Missing values found:")
        logger.info(missing_report)
    
    # Handle missing values in 'internet_type'
    df['internet_type'] = df['internet_type'].apply(lambda x: 'no_internet' if pd.isna(x) else x)
    
    # Handle missing values in 'offer'
    df['offer'] = df['offer'].apply(lambda x: 'no_offer' if pd.isna(x) else x)
    
    # Drop columns with missing values that would cause data leakage
    df = df.drop(['churn_category', 'churn_reason'], axis=1)

    logger.debug(f"Missing values after handling: {missing_values(df)}")
    
    return df

def remove_redundant_features(df):
    """Remove redundant and unnecessary features"""
    logger.info("Removing redundant features...")
    
    # Check and remove 'internet_service' as it's redundant with 'internet_type'
    internet_correspondence = ((df['internet_service'] == 'No') == (df['internet_type'] == 'no_internet')).all()
    logger.debug(f"'internet_service' corresponds perfectly to 'internet_type': {internet_correspondence}")
    
    # Check and remove 'dependents' as it's redundant with 'number_of_dependents'
    dependents_correspondence = ((df['dependents'] == 'No') == (df['number_of_dependents'] == 0)).all()
    logger.debug(f"'dependents' corresponds perfectly to 'number_of_dependents': {dependents_correspondence}")
    
    # Check and remove 'referred_a_friend' as it's redundant with 'number_of_referrals'
    referrals_correspondence = ((df['referred_a_friend'] == 'No') == (df['number_of_referrals'] == 0)).all()
    logger.debug(f"'referred_a_friend' corresponds perfectly to 'number_of_referrals': {referrals_correspondence}")
    
    # Remove features with no variance, identifier features, and location-related features
    cols_to_drop = [
        # Redundant features just analyzed:
        'internet_service', 'dependents', 'referred_a_friend',

        # No variance/identifier features
        'count', 'country', 'state', 'quarter', 'id', 'location_id', 'status_id', 'service_id',
        
        # Location features
        'latitude', 'longitude', 'lat_long', 'city', 'zip_code', 'population',
        
        # Age-related features (keep only 'age')
        'under_30', 'senior_citizen',
        
        # Churn-related features that cause data leakage (keep only 'churn_value')
        'churn_label', 'churn_score', 'customer_status'
    ]
    
    df = df.drop(cols_to_drop, axis=1)
    logger.debug(f"df shape after removing redundant features: {df.shape}")
    return df

def cardinality(df, max_display=3):
    """Calculate cardinality of each col and provide samples"""
    res = pd.DataFrame({
        'nunique': df.nunique(),
        'dtype': df.dtypes
    }).sort_values(by='nunique')

    res['unique_values'] = [
        df[col].unique().tolist() if res.loc[col, 'nunique'] <= max_display
        else f"[{', '.join(map(str, df[col].unique()[:max_display]))} ...(cont'd)]"
        for col in res.index
    ]

    res['missing_value_count'] = [
        df[col].isna().sum() for col in df.columns
    ]

    return res

def encode_binary_features(df):
    """Encode binary categorical features to 0/1"""
    logger.info("Encoding binary features...")
    
    # Get binary features
    binary_features = cardinality(df)[cardinality(df)['nunique'] == 2].index.to_list()
    cols_to_encode = [c for c in binary_features if c not in ['gender', 'churn_value']]
    
    # Encode to 0/1
    for col in cols_to_encode:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
    
    logger.debug(f"Columns encoded: {cols_to_encode}")
    return df

# === Example of object-oriented figure handling ===
def plot_correlation_matrix(df, figsize=(20, 20)):
    """Plot correlation matrix for numerical features"""
    logger.info("Plotting correlations...")
    fig, ax = plt.subplots(figsize=figsize)
    correlation = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'correlation_matrix.png')
    plt.close(fig)

def analyze_correlations(df):
    """Analyze correlations and handle highly correlated features"""
    logger.info("Analyzing correlations...")
    
    # Drop 'total_charges' as it's strongly correlated with 'total_revenue'
    df = df.drop(['total_charges'], axis=1)
    
    # Create a new streaming feature to replace 'streaming_movies' and 'streaming_music' that is 1 if both are 1, 0 otherwise
    movies_music_corr = (df['streaming_movies'] == df['streaming_music']).value_counts(normalize=True)
    logger.debug(f"Correspondence between 'streaming_movies' and 'streaming_music': {movies_music_corr[True]:.2%}")
    
    df['streaming'] = df['streaming_movies'] * df['streaming_music']
    df = df.drop(['streaming_movies', 'streaming_music'], axis=1)
    
    return df

def check_satisfaction_score(df):
    """Check if satisfaction_score has data leakage issues"""
    logger.info("Checking satisfaction_score for data leakage...")
    
    # Check if satisfaction score perfectly predicts churn
    low_score_churn = df[df['satisfaction_score'] < 3]['churn_value'].value_counts(normalize=True)
    high_score_churn = df[df['satisfaction_score'] > 3]['churn_value'].value_counts(normalize=True)
    
    # Report findings
    if 1 in low_score_churn:
        logger.debug(f"Customers with satisfaction_score < 3 churn rate: {low_score_churn[1]:.2%}")
    if 0 in high_score_churn:
        logger.debug(f"Customers with satisfaction_score > 3 non-churn rate: {high_score_churn[0]:.2%}")
    
    # Drop satisfaction_score if it's causing data leakage
    if ((1 in low_score_churn and low_score_churn[1] > 0.95) or 
        (0 in high_score_churn and high_score_churn[0] > 0.95)):
        logger.debug("Data leakage detected: satisfaction_score is too predictive of churn")
        df = df.drop(['satisfaction_score'], axis=1)
    
    return df

def transform_skewed_features(df):
    """Apply log transformation to heavily skewed numerical features"""
    logger.info("Transforming skewed features...")
    
    # Apply log transform to right-skewed features
    for col in ['total_long_distance_charges', 'total_revenue']:
        logger.debug(f"Skewness of {col}: {df[col].skew()}")
        df[f"log_{col}"] = np.log1p(df[col])
        logger.debug(f"After log-transform: {np.log1p(df[col]).skew()}")
    
    # Drop original skewed features
    df = df.drop(['total_long_distance_charges', 'total_revenue'], axis=1)
    
    return df

def plot_feature_distributions(df, target_col='churn_value', cat_cutoff=10):
    """
    Plot feature distributions and their relationship with the target
    
    Features with cardinality less than cat_cutoff are considered categorical
    """
    logger.info("Plotting feature distributions...")

    # Target distribution
    fig_target, ax = plt.subplots(figsize=(6, 6))
    s = sns.countplot(data=df, y=target_col, ax=ax)

    # Calculate percentages
    total = len(df)
    counts = df[target_col].value_counts(sort=False).values
    percentages = counts / total * 100

    labels = [f"{count} ({percentage:.1f}%)" for count, percentage in zip(counts, percentages)]
    ax.bar_label(s.containers[0], labels=labels, label_type='center')

    ax.set_title(f"Distribution of {target_col}")
    plt.tight_layout()
    fig_target.savefig(FIGURES_DIR / 'target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig_target)

    # Get categorical features
    card = cardinality(df)
    cat_cols = card[card['nunique'] < cat_cutoff].index.to_list()

    # Remove target
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    
    logger.debug(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    
    # Plot categorical features
    fig_cat, axes = plt.subplots(len(cat_cols), 3, figsize=(20, len(cat_cols) * 3))
    
    for i, col in enumerate(cat_cols):
        cat_order = sorted(df[col].unique())
        
        # Distribution of col
        sns.countplot(data=df, x=col, ax=axes[i, 0], hue=col, palette='Dark2', order=cat_order)
        axes[i, 0].set_title(f"Distribution of {col}")
        axes[i, 0].set_xlabel('')
        axes[i, 0].set_ylabel('Count')

        # Distribution of col by target
        sns.countplot(data=df, x=col, hue=target_col, ax=axes[i, 1], order=cat_order)
        axes[i, 1].legend(title=target_col, loc='upper right')
        axes[i, 1].set_title(f"{col} by {target_col}")
        axes[i, 1].set_xlabel('')
        axes[i, 1].set_ylabel('Count')

        # Percentage of target within each value of col
        churn_by_cat = df.groupby(col)[target_col].mean() * 100
        churn_by_cat = churn_by_cat.reset_index()
        sns.barplot(data=churn_by_cat, x=col, y=target_col, ax=axes[i, 2], 
                    hue=col,
                    palette='Dark2', order=cat_order)
        axes[i, 2].set_title(f"{target_col} rate by {col}")
        axes[i, 2].set_xlabel('')
        axes[i, 2].set_ylabel('Percentage')
        axes[i, 2].set_ylim(0, churn_by_cat[target_col].max() * 1.1)
        
        for p in axes[i, 2].patches:
            axes[i, 2].annotate(f"{p.get_height():.1f}%",
                            (p.get_x() + p.get_width() / 2, p.get_height()),
                            ha='center', va='bottom')
    
    plt.suptitle("Distribution of Categorical Features", y=1)
    plt.tight_layout()
    fig_cat.savefig(FIGURES_DIR / 'categorical_features.png', dpi=300, bbox_inches='tight')
    plt.close(fig_cat)

    logger.info(f"Categorical features distribution plot added to {FIGURES_DIR}")
    
    # Get numerical features
    num_cols = card[card['nunique'] >= cat_cutoff].index.to_list()

    logger.debug(f"Numerical features ({len(num_cols)}): {num_cols}")
    
    # Plot numerical features
    fig_num, axes = plt.subplots(len(num_cols), 2, figsize=(12, len(num_cols) * 3))
    
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, ax=axes[i, 0], kde=True)
        axes[i, 0].set_title(f"Distribution of {col}")
        axes[i, 0].set_xlabel('')

        sns.boxplot(data=df, x=target_col, y=col, ax=axes[i, 1])
        axes[i, 1].set_title(f"{col} by {target_col}")
        axes[i, 1].set_xlabel('')
        axes[i, 1].set_ylabel('')
    
    plt.suptitle("Distribution of Numerical Features", y=1)
    plt.tight_layout()
    fig_num.savefig(FIGURES_DIR / 'numerical_features.png', dpi=300, bbox_inches='tight')
    plt.close(fig_num)

    logger.info(f"Numerical features distribution plot added to {FIGURES_DIR}")

def run_data_cleaning(plot_check=False):
    """
    Main function to load, process and sace cleaned dataset.
    Returns True if successful, False otherwise.
    """
    # Ensure the figures dirctory exists
    create_directory(FIGURES_DIR)

    # Load data
    df = load_data()
    logger.info(f"Initial shape: {df.shape}")

    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove redundant features
    df = remove_redundant_features(df)
    
    # Encode binary features
    df = encode_binary_features(df)
    
    # Analyze correlations and handle highly correlated features
    if plot_check:
        plot_correlation_matrix(df)
    df = analyze_correlations(df)

    # Check satisfaction_score for data leakage
    df = check_satisfaction_score(df)
    
    # Transform skewed features
    df = transform_skewed_features(df)
    
    # Visualize features
    if plot_check:
        plot_feature_distributions(df)
    
    # logger.info dataset info
    logger.info(f"Final shape: {df.shape}")
    logger.info("Feature cardinality:")
    logger.info(cardinality(df))

    # Save cleaned dataset
    output_path = PROCESSED_DATA_DIR / 'telco_cleaned.csv'
    df.to_csv(output_path)
    logger.info(f"Cleaned dataset saved to {output_path}")
    return True

def main():
    return run_data_cleaning()

if __name__ == "__main__":
    success = main()
    if not success:
        logger.warning("Data cleaning script completed with errors. Review output and check notebooks.")
    else:
        logger.info("Data cleaning script completed successfully.")