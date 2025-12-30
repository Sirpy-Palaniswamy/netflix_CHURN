# Netflix Churn Analysis
- This project builds an end-to-end machine learning pipeline to predict customer churn for a subscription-based digital service.
- The goal is to identify at-risk users early using behavioral, engagement, and subscription features.

###### ※ Report Findings are summarized in the `business_report` section of the repository

## Problem Statement
Customer churn is a critical business problem for subscription-based services.
This project aims to identify users likely to churn by analyzing behavioral,
engagement, and subscription-related data.

The goal is not only prediction accuracy, but also interpretability and
business-driven insights that can support retention strategies.
#
**Key objectives:**
- Analyze user behavior patterns linked to churn
- Engineer meaningful engagement features
- Train and evaluate regression model
- Interpret model outputs for business analytics report

## Dataset Overview:
[Netflix-Style Synthetic Dataset]([url](https://www.kaggle.com/datasets/sayeeduddin/netflix-2025user-behavior-dataset-210k-records/data)) 
- **Dataset Overview** - This synthetic dataset simulates a Netflix-style streaming platform with realistic user behavior, content catalog, and engagement metrics. Perfect for machine learning, data science, and analytics projects.

- **Key Features of `"users.csv"`** - 10,300 total records with User demographics, subscription plans, regional data
- **Key Features of `"watch_history.csv"`**	- 105,000 total records	with viewing sessions with device, quality, progress data
- **Key Features**
- - The Final pruned dataset consists of:
  - Size: 8,431 users
  - 17 engineered features
  - Target Variable:
  - - churn (1 = churned, 0 = retained)
## Feature Groups
| Category           | Features                                           |
| ------------------ | -------------------------------------------------- |
| Demographics       | age, gender, country                               |
| Subscription       | subscription_plan, monthly_spend                   |
| Engagement         | avg_session_minutes, num_sessions, watch_per_day   |
| Retention Signals  | tenure_days, days_since_last_watch                 |
| Content Completion | completion_rate                                    |
| Engineered Buckets | watch_bucket, inactivity_bucket, completion_bucket |
| Value Metrics      | value_ratio                                        |
#
## Feature Engineering

Key engineered features include:

- **Engagement intensity**
  - `watch_per_day`
  - `avg_session_minutes`

- **Inactivity modeling**
  - `days_since_last_watch`
  - `inactivity_bucket`

- **Consumption quality**
  - `completion_rate`
  - `completion_bucket`

- **Customer value**
  - `value_ratio` (watch time relative to spend)

These features are designed to capture early disengagement patterns rather than relying solely on static demographic attributes.
## Modeling Approach
- Model: Logistic Regression
- Reason:
  - Strong baseline for churn problems
  - Interpretable coefficients
  - Easy communication with business stakeholders

## Model Evaluation
### Classification Performance

| Metric | Value |
|------|------|
| Accuracy | 0.50 |
| ROC-AUC | 0.49 |
| Recall (Churn) | 0.49 |
| Precision (Churn) | 0.14 |

The model captures nearly half of churned users (high recall) but produces many false positives, indicating weak linear separability in the data.

##  Visual Diagnostics

### ROC Curve
![ROC Curve](visuals/ROC_Curve.png)

### Confusion Matrix
![Confusion Matrix](visuals/confusion_matrix.png)

### Feature Coefficients
![Coefficients](visuals/Top_Churn_Drivers.png)

## Business Interpretation
- Suitable for early churn warning systems
- High recall supports proactive retention campaigns
- False positives are acceptable in marketing-driven interventions
- Indicates need for nonlinear or temporal models for better discrimination

## Limitations & Future Work

- Logistic regression struggles with nonlinear behavior patterns
- Class imbalance affects precision
- Temporal behavior trends are not explicitly modeled

Future improvements:
- Gradient boosting models (XGBoost, LightGBM)
- Cost-sensitive learning
- Time-series engagement features
- SHAP-based explainability

## Frameworks
- Python,
- Pandas,
- Scikit-learn,
- Matplotlib,
- Seaborn,
- Jupyter Notebook
