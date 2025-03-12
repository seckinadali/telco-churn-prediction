# Telco Customer Churn Prediction

## Project Overview
This repository contains a machine learning project focused on predicting customer churn for a telecommunications company. Customer churn, or the rate at which customers stop doing business with a company, is a critical metric that directly impacts revenue. This project aims to identify factors that contribute to churn and build predictive models to help proactively identify at-risk customers.

## Current Status
This project is currently in development. The following components have been completed:

- **Data Preparation**: Loading, cleaning, and merging multiple datasets from [IBM's sample telco data](https://accelerator.ca.analytics.ibm.com/bi/?perspective=authoring&pathRef=.public_folders%2FIBM%2BAccelerator%2BCatalog%2FContent%2FDAT00148&id=i9710CF25EF75468D95FFFC7D57D45204&objRef=i9710CF25EF75468D95FFFC7D57D45204&action=run&format=HTML&cmPropStr=%7B%22id%22%3A%22i9710CF25EF75468D95FFFC7D57D45204%22%2C%22type%22%3A%22reportView%22%2C%22defaultName%22%3A%22DAT00148%22%2C%22permissions%22%3A%5B%22execute%22%2C%22read%22%2C%22traverse%22%5D%7D), which include customer demographics, service usage details and churn information
- **Feature Engineering**: Handling missing values, identifying and removing redundant features, analyzing correlations, and transforming skewed features

Upcoming work includes:
- Model development and selection
- Hyperparameter tuning and cross-validation
- Model evaluation and interpretation
- Deployment considerations and business recommendations

## Repository Structure
```
telco-churn-prediction/
├── data/
│   ├── raw/              # Original Excel files (not included in repository)
│   └── processed/        # Cleaned CSV files
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_feature_engineering.ipynb
├── README.md
└── .gitignore
```

## Key Insights So Far
- Month-to-month contracts are strongly associated with higher churn rates
- Customers with fiber optic internet service are more likely to churn
- Tenure is a significant predictor - newer customers are at higher risk
- Certain payment methods (particularly bank withdrawal) correlate with increased churn