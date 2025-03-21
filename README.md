# Telco Customer Churn Prediction

## Project Overview
This repository contains a machine learning project focused on predicting customer churn for a telecommunications company. Customer churn, or the rate at which customers stop doing business with a company, is a critical metric that directly impacts revenue. This project aims to identify factors that contribute to churn and build predictive models to help proactively identify at-risk customers.

## Current Status
This project is a work in progress with most of the ML pipeline implemented. The following components have been completed:

- **Data Collection & Cleaning**: Loading, cleaning, and merging multiple datasets from [IBM's sample telco data](https://accelerator.ca.analytics.ibm.com/bi/?perspective=authoring&pathRef=.public_folders%2FIBM%2BAccelerator%2BCatalog%2FContent%2FDAT00148&id=i9710CF25EF75468D95FFFC7D57D45204&objRef=i9710CF25EF75468D95FFFC7D57D45204&action=run&format=HTML&cmPropStr=%7B%22id%22%3A%22i9710CF25EF75468D95FFFC7D57D45204%22%2C%22type%22%3A%22reportView%22%2C%22defaultName%22%3A%22DAT00148%22%2C%22permissions%22%3A%5B%22execute%22%2C%22read%22%2C%22traverse%22%5D%7D), including customer demographics, service usage details, and churn information, into a single consolidated file
- **Exploratory Data Analysis & Feature Engineering**: Handling missing values, identifying and removing redundant features, transforming features as needed, exploring data distributions and correlations, and analyzing feature relationships with the target variable
- **Model Development**: Creating preprocessing and modeling pipelines, comparing baseline models with cross-validation, handling class imbalance, initial hyperparameter tuning to select the most promising models for different business objectives

## Modeling Approach

Three business perspectives are explored:

- **Balanced approach**: Optimizing for F1-score to balance precision and recall

- **Precision-focused approach**: Prioritizing accurate identification of true churners (minimizing false positives)

- **Recall-focused approach**: Prioritizing identification of as many potential churners as possible (minimizing false negatives)

A two-stage hyperparameter tuning process is applied:

1. **Initial Light Tuning**: Perform a lightweight hyperparameter search on all models to get a better sense of their potential.

2. **Final Deep Tuning**: Select the most promising models for each business objective based on the initial tuning results, then perform a more extensive hyperparameter search.

#### Models evaluated:

- Logistic Regression
- Linear SVM
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- K-Nearest Neighbors
- Naive Bayes

#### Class imbalance techniques applied:

- Class weights
- SMOTE (Synthetic Minority Over-sampling Technique)

#### Work in Progress

- Extensive hyperparameter search for each business objective
- Feature importance analysis
- Model performance summary

## Repository Structure
```
telco-churn-prediction/
├── data/
│   ├── raw/              # Original Excel files (not included in repository)
│   └── processed/        # Cleaned CSV files
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_development.ipynb (partially complete)
├── README.md
└── .gitignore
```

## Data Source
[IBM's Sample Datasets](https://accelerator.ca.analytics.ibm.com/bi/?perspective=authoring&pathRef=.public_folders%2FIBM%2BAccelerator%2BCatalog%2FContent%2FDAT00148&id=i9710CF25EF75468D95FFFC7D57D45204&objRef=i9710CF25EF75468D95FFFC7D57D45204&action=run&format=HTML&cmPropStr=%7B%22id%22%3A%22i9710CF25EF75468D95FFFC7D57D45204%22%2C%22type%22%3A%22reportView%22%2C%22defaultName%22%3A%22DAT00148%22%2C%22permissions%22%3A%5B%22execute%22%2C%22read%22%2C%22traverse%22%5D%7D)
