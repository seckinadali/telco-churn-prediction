# Telco Customer Churn Prediction

## Project Overview
This repository contains a machine learning project focused on predicting customer churn for a telecommunications company. Customer churn, or the rate at which customers stop doing business with a company, is a critical metric that directly impacts revenue. This project aims to identify factors that contribute to churn and build predictive models to help proactively identify at-risk customers.

## Current Status
This project is a work in progress with most of the ML pipeline implemented. The following components have been completed:

- **Data Preparation**: Loading, cleaning, and merging multiple datasets from [IBM's sample telco data](https://accelerator.ca.analytics.ibm.com/bi/?perspective=authoring&pathRef=.public_folders%2FIBM%2BAccelerator%2BCatalog%2FContent%2FDAT00148&id=i9710CF25EF75468D95FFFC7D57D45204&objRef=i9710CF25EF75468D95FFFC7D57D45204&action=run&format=HTML&cmPropStr=%7B%22id%22%3A%22i9710CF25EF75468D95FFFC7D57D45204%22%2C%22type%22%3A%22reportView%22%2C%22defaultName%22%3A%22DAT00148%22%2C%22permissions%22%3A%5B%22execute%22%2C%22read%22%2C%22traverse%22%5D%7D), including customer demographics, service usage details, and churn information, into a single consolidated file
- **Feature Engineering**: Handling missing values, identifying and removing redundant features, transforming features as needed, exploring data distributions and correlations, and analyzing feature relationships with the target variable
- **Model Development**: Creating preprocessing and modeling pipelines, comparing baseline models with cross-validation, handling class imbalance, hyperparameter tuning for selected models and evaluating models on test data with various metrics

In progress:
- Feature importance analysis
- Model performance summary

Upcoming work:
- Implementing a proper MLOps workflow
- Deployment considerations and business recommendations

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

## Technologies Used
- Python (Pandas, NumPy)
- Scikit-learn
- XGBoost
- Matplotlib & Seaborn
- Imbalanced-learn

## Model Development
The project is implementing several models optimized for different business objectives:
- Logistic Regression (optimized for F1-score)
- Random Forest (optimized for precision)
- Linear SVM (optimized for recall)

Class imbalance is being addressed using balanced class weights.

## Current Results
Preliminary model performance (subject to further optimization):
- A balanced model for general churn prediction (F1-score: ~0.68)
- A precision-focused model for targeted interventions (Precision: ~0.86)
- A recall-focused model to minimize missed potential churners (Recall: ~0.86)

## Future Work
- Complete feature importance analysis
- Develop proper MLOps workflow for model versioning and monitoring
- Create visualizations for stakeholders
- Deploy the model in a production environment

## Data Source
[IBM's Sample Datasets](https://accelerator.ca.analytics.ibm.com/bi/?perspective=authoring&pathRef=.public_folders%2FIBM%2BAccelerator%2BCatalog%2FContent%2FDAT00148&id=i9710CF25EF75468D95FFFC7D57D45204&objRef=i9710CF25EF75468D95FFFC7D57D45204&action=run&format=HTML&cmPropStr=%7B%22id%22%3A%22i9710CF25EF75468D95FFFC7D57D45204%22%2C%22type%22%3A%22reportView%22%2C%22defaultName%22%3A%22DAT00148%22%2C%22permissions%22%3A%5B%22execute%22%2C%22read%22%2C%22traverse%22%5D%7D)
