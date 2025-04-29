# Telco Customer Churn Prediction

A machine learning pipeline to predict customer churn for a telecommunications company using data from IBM's sample datasets.

## Project Overview

This project develops a predictive model for customer churn in a telecommunications company. Through data preparation, exploratory analysis, feature engineering, and model development, we identify key factors influencing customer decisions to leave the company, and create models that can predict potential churners with different optimization goals.

### Key Features

- Complete end-to-end machine learning pipeline with detailed documentation
- Data preprocessing workflows including merging, cleaning, and feature engineering
- Extensive exploratory data analysis with visualizations
- Model development with threshold tuning for different business objectives
- Feature importance analysis with multiple techniques (XGBoost, SHAP, Linear SVC)

## Development Approach

This project follows a deliberate two-phase development methodology:

### 1. Exploration & Analysis (Notebooks)
The Jupyter notebooks contain all experimental work including:
- Data exploration and visualization
- Multiple model comparisons and class imbalance techniques
- Hyperparameter tuning experiments
- Detailed visualizations and rationale for decisions

### 2. Production Implementation (Modular Code)
The Python modules implement only the final, optimized approaches:
- Focused on the best-performing models (XGBoost and Linear SVC)
- Streamlined preprocessing with decisions already made
- Efficient implementation without experimental components
- Production-ready code with proper logging and error handling

This separation creates a codebase that maintains both comprehensive documentation of the analysis process (notebooks) and clean, maintainable implementation code (modules).

## Data

The project uses IBM's sample datasets for telecom customer churn, which include:
- Customer demographics
- Service information
- Account details
- Location data
- Churn status

## Modeling Approach

We developed multiple models optimized for different business objectives:

### Business Perspectives
- **Balanced approach**: Optimizing for F1-score to balance precision and recall
- **Precision-focused approach**: Prioritizing accurate identification of true churners (minimizing false positives)
- **Recall-focused approach**: Prioritizing identification of as many potential churners as possible (minimizing false negatives)

### Models Evaluated
- Logistic Regression
- Linear SVM
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- K-Nearest Neighbors
- Naive Bayes

### Class Imbalance Techniques
- Class weights
- SMOTE (Synthetic Minority Over-sampling Technique)

### Model Performance

| Model | F1 Score | Precision | Recall | Accuracy | Business Use Case |
|-------|----------|-----------|--------|----------|-------------------|
| XGBoost (Default 0.5) | 0.70 | 0.61 | 0.82 | 0.81 | General purpose |
| XGBoost (Optimal F1) | 0.69 | 0.70 | 0.69 | 0.84 | Balanced approach |
| XGBoost (High Recall) | 0.68 | 0.54 | 0.90 | 0.77 | Minimize missed churners |
| XGBoost (High Precision) | 0.55 | 0.84 | 0.40 | 0.82 | Minimize false alarms |
| Linear SVC | 0.66 | 0.68 | 0.64 | 0.82 | Simple, interpretable model |

*Note: Our F1-optimized model happened to produce nearly identical precision and recall values on our dataset, though generally F1 optimization doesn't guarantee equal metrics.*

### Implementation Note

The metrics above reflect our current production code results. When comparing with the notebook exploration, we see minor differences: XGBoost metrics differ by at most 1.2 percentage points, while the Linear SVC model shows almost identical results (differences of less than 0.004 percentage points).

These small discrepancies (typically under 1%) arise from random seeds, implementation details, and execution environments - normal variations in machine learning systems. The stability of Linear SVC between implementations highlights how linear models tend to be less sensitive to these factors compared to complex ensemble methods like XGBoost.

## Key Findings

Our analysis identified several critical factors influencing customer churn:

1. **Contract Type**: Customers on month-to-month contracts are significantly more likely to churn compared to those with one or two-year commitments.

2. **Referral Behavior**: Number of referrals strongly correlates with reduced churn probability, suggesting satisfied customers who refer others are much less likely to leave.

3. **Monthly Charges**: Higher monthly charges increase churn risk, highlighting price sensitivity.

4. **Tenure**: New customers show higher churn rates, with the first 12 months being a critical period for retention.

5. **Internet Service Type**: Fiber optic service shows a complex relationship with churn that requires deeper investigation.

6. **Household Composition**: Accounts with dependents exhibit lower churn rates, suggesting family-oriented services create stronger customer relationships.

## Repository Structure

```
telco-churn-prediction/
├── data/
│   ├── raw/        # Original Excel files (not included in repository)
│   └── processed/  # Cleaned CSV files
├── figures/
├── models/
├── notebooks/
│   ├── 01_data_merging.ipynb
│   ├── 02_data_cleaning_and_eda.ipynb
│   └── 03_modeling_and_evaluation.ipynb
├── logs/           # Execution logs (not included in repository)
├── src/
│   ├── data_merging.py
│   ├── data_cleaning.py
│   ├── model_development.py
│   ├── feature_importance.py
│   └── __init__.py
├── main.py
├── logger.py       # Logging configuration
└── README.md
```

## Getting Started

### Running the Pipeline

To run the complete pipeline:
```
python main.py
```

Or run individual components:
```
python src/data_merging.py
python src/data_cleaning.py
python src/model_development.py
python src/feature_importance.py
```

### Notebooks

For detailed analysis and results:

1. **01_data_merging.ipynb**: Consolidates multiple Excel files into a single dataset
2. **02_data_cleaning_and_eda.ipynb**: Cleans data, handles missing values, and performs exploratory analysis
3. **03_modeling_and_evaluation.ipynb**: Develops predictive models, optimizes thresholds, and analyzes feature importance

## Conclusion

This project developed a churn prediction framework with models tailored to different business objectives. The dynamic threshold adjustment approach provides flexibility without retraining models, addressing the imbalanced nature of the dataset effectively.

**Limitations**:
- The analysis identifies factors associated with churn, but not necessarily causal relationships
- Due to the nature of the data, uses a static approach without time-series data to capture behavior over time

**Future Work**:
- Cost-benefit analysis for the impact of retention strategies
- Designing controlled experiments to apply A/B testing on the effectiveness of specific interventions

## Data Source
[IBM's Sample Datasets](https://accelerator.ca.analytics.ibm.com/bi/?perspective=authoring&pathRef=.public_folders%2FIBM%2BAccelerator%2BCatalog%2FContent%2FDAT00148&id=i9710CF25EF75468D95FFFC7D57D45204&objRef=i9710CF25EF75468D95FFFC7D57D45204&action=run&format=HTML&cmPropStr=%7B%22id%22%3A%22i9710CF25EF75468D95FFFC7D57D45204%22%2C%22type%22%3A%22reportView%22%2C%22defaultName%22%3A%22DAT00148%22%2C%22permissions%22%3A%5B%22execute%22%2C%22read%22%2C%22traverse%22%5D%7D)