Telco Customer Churn Prediction & Retention Strategy
End-to-end telecom customer churn analysis, machine learning pipeline, SHAP explainability, and interactive Streamlit dashboard built with Python, SQLite, Scikit-Learn, XGBoost, and SHAP.

Key Highlights
Dataset & SQL Analysis: Analyzed 3,333 telecom customer records (21 base attributes) in SQLite, uncovering a 14.49% overall churn rate that rises to 42.41% for international plan customers and exceeds 45–63% for customers with 4+ service calls.

Feature Engineering: Engineered 9 domain-specific features (total_minutes, total_charge, avg_charge_per_minute, intl_plan_flag, high_service_calls_flag, and related usage/billing indicators) with a stratified train/test split.

Model Performance: Compared baseline Logistic Regression (0.8522 ROC-AUC) against XGBoost, achieving 96.55% accuracy, 0.9744 precision, 0.7835 recall, 0.8686 F1-score, and 0.8966 ROC-AUC on the held-out test set.

Explainability & App: Generated global SHAP visualizations highlighting total_charge, intl_plan_flag, and customer_service_calls as top churn drivers, and built a Streamlit dashboard (app.py) for live churn scoring, cohort exploration, and retention recommendations.

Project Structure
app.py — Interactive Streamlit dashboard (Live Churn Scorer, SQL & Cohort Insights, and SHAP Explainability)

src/load_to_sql.py — Loads dataset into SQLite database (data/churn.db)

src/run_sql.py & sql/ — Executes SQL queries for cohort and churn rate analysis

src/features.py — Builds engineered features and prepares train/test datasets

src/train.py — Trains Logistic Regression and XGBoost models and logs evaluation metrics

src/explain.py — Computes SHAP values and saves reports/shap_summary.png

reports/shap_summary.png — Global SHAP feature importance summary plot

How to Run Locally
Install dependencies:

Bash
pip install pandas numpy scikit-learn xgboost shap streamlit joblib
Run the pipeline scripts in order (if artifacts are not already present):

Bash
python src/load_to_sql.py
python src/run_sql.py
python src/features.py
python src/train.py
python src/explain.py
Launch the Streamlit dashboard:

Bash
streamlit run app.py
Author
Shivraj — B.Sc. in Computer Science (Raichur University / Government First Grade College, Shorapur)
GitHub: github.com/shivv7460-dot
