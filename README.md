# Explainable AI for Healthcare Prediction Models: Diabetes Prediction

A machine learning project that predicts diabetes risk in patients while making the model's reasoning transparent and interpretable. The aim is not just accurate prediction but trustworthy prediction, so that the factors driving each decision can be inspected, explained, and validated against clinical knowledge.

## Overview

Healthcare prediction models are often treated as black boxes, which limits their adoption in settings where clinicians need to understand and trust a recommendation before acting on it. This project pairs standard classification models with two explainability frameworks, SHAP and LIME, to surface both global feature importance across the whole dataset and local explanations for individual patient predictions.

## Dataset

The project uses the Pima Indians Diabetes Dataset, originally from the UCI Machine Learning Repository and widely available on Kaggle. It contains 768 records with eight clinical features and a binary outcome indicating diabetes diagnosis.

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age in years |
| Outcome | Target variable (1 = diabetes, 0 = no diabetes) |

## Methodology

The pipeline moves from raw data through cleaning, feature work, modeling, and finally explainability.

**Data cleaning.** Several columns contain physiologically impossible zero values that represent missing data rather than true readings. These are imputed with column medians for Glucose, BloodPressure, SkinThickness, Insulin, and BMI. Outliers are capped using the interquartile range method, and all numerical features are scaled to a common range with MinMax normalization.

**Feature engineering.** The work adds a log transform of Insulin to reduce skewness, an interaction term between BMI and Glucose, and an age category derived from the Age column. Correlation analysis is used to identify the features most associated with the outcome.

**Modeling.** Two classifiers are trained and compared, Logistic Regression and Random Forest. The Random Forest is further refined through a grid search over the number of estimators, tree depth, and the minimum samples required to split or form a leaf, with ROC-AUC as the optimization target.

**Explainability.** SHAP provides both a global summary of which features drive predictions across the dataset and a force plot breaking down a single prediction. A comparative analysis contrasts the average feature contributions for diabetic versus non-diabetic predictions, and a focused look at Glucose checks whether the model's behavior aligns with clinical expectations. LIME supplements this with an interpretable, locally faithful explanation of individual predictions.

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.75 | 0.83 |
| Random Forest | 0.77 | 0.83 |
| Tuned Random Forest | 0.77 | 0.84 |

Performance is evaluated with classification reports, confusion matrices, and combined ROC and Precision-Recall curves that place all three models side by side. The tuned Random Forest gives the strongest overall ranking ability while keeping the model interpretable through SHAP and LIME.

## Repository Structure

| File | Description |
|---|---|
| `diabetes_prediction.ipynb` | Main notebook covering the full workflow from loading to explainability |
| `streamlit_app.py` | Streamlit application for interactive exploration |
| `data/diabetes.csv` | Raw Pima Indians Diabetes dataset |
| `data/diabetes_cleaned.csv` | Processed dataset after cleaning and feature engineering |
| `README.md` | Project documentation |

```
diabetes_prediction.ipynb
streamlit_app.py
README.md
data/
    diabetes.csv
    diabetes_cleaned.csv
```

## Getting Started

Clone the repository and install the dependencies. The core libraries are pandas, numpy, matplotlib, seaborn, scikit-learn, shap, and lime.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap lime streamlit
```

To run the notebook:

```bash
jupyter notebook diabetes_prediction.ipynb
```

To launch the interactive app:

```bash
streamlit run streamlit_app.py
```

## Key Takeaways

The project shows that a model can be both reasonably accurate and transparent. Glucose emerges as a leading predictor of diabetes risk, which matches established clinical understanding and gives confidence that the model is learning meaningful patterns rather than spurious correlations. By exposing these contributions at both the population and individual level, the work demonstrates how explainability turns a prediction into something a clinician can reason about and trust.