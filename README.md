# Bachelor Thesis: Machine Learning for COPD Exacerbation Forecasting

Machine-learning pipeline for forecasting the daily occurrence of emergency department visits due to **chronic obstructive pulmonary disease (COPD) exacerbations** using air-pollution, meteorological and temporal variables.

This repository is a refactored and modularised version of the code developed for my **Bachelor Thesis in Biomedical Engineering**. The original analysis has been reorganised into a reproducible Python project with reusable modules, sequential notebooks, automated tests and stored model results.

## Overview

COPD exacerbations are clinically relevant events associated with increased emergency department utilisation, hospitalisation and disease burden.

The objective of this project was to investigate whether environmental and temporal information could help predict whether **at least one COPD-related emergency department visit would occur on a given day**.

The study used aggregated daily observations from the **Valladolid Oeste Health Area, Spain**, covering the period **2010–2023**.

The modelling workflow combines:

- air-pollution variables;
- meteorological variables;
- immediate, accumulated and lagged exposures;
- seasonal and calendar information;
- temporal feature selection;
- machine-learning classification;
- temporal cross-validation;
- independent test evaluation;
- explainable AI using SHAP.

## Problem Definition

The original daily count of COPD-related emergency department visits was transformed into a binary classification target:

- `0`: no COPD-related emergency department visit;
- `1`: at least one COPD-related emergency department visit.

Two alternative predictor representations were evaluated.

### Subset A — Immediate and accumulated exposure

Environmental variables were represented using:

- immediate exposure;
- cumulative exposure from day 0 to day 3;
- cumulative exposure from day 0 to day 7;
- temporal predictors.

Initial size: **48 predictors**

### Subset B — Individual temporal lags

Environmental variables were represented using individual values from:

- lag 0;
- lag 1;
- ...
- lag 7;
- temporal predictors.

Initial size: **155 predictors**

## Machine-Learning Pipeline

The complete workflow is organised as:

```text
Raw daily data
      │
      ▼
Data quality assessment
      │
      ▼
Cleaning and temporal interpolation
      │
      ▼
Feature engineering
      │
      ├── Subset A: immediate + accumulated exposures
      │
      └── Subset B: individual lags 0–7
      │
      ▼
Temporal train/test split
      │
      ├── Development: 2010–2022
      └── Independent test: 2023
      │
      ▼
Correlation filtering
      │
      ▼
mRMR + temporal block bootstrap
      │
      ▼
Model development
      │
      ├── LightGBM
      ├── CatBoost
      └── Multilayer Perceptron
      │
      ▼
Temporal cross-validation
      │
      ├── Sliding window
      └── Expanding window
      │
      ▼
Final evaluation on 2023
      │
      ▼
SHAP explainability
```

## Feature Selection

Feature selection combined:

1. correlation filtering to reduce multicollinearity;
2. minimum Redundancy Maximum Relevance (**mRMR**);
3. **1,000 temporal block-bootstrap replications**;
4. selection-frequency thresholds derived from the bootstrap distribution.

The final predictor sets contained:

| Predictor set | Initial | After correlation filtering | Final selected |
|---|---:|---:|---:|
| Subset A | 48 | 24 | 6 |
| Subset B | 155 | 76 | 24 |

NO₂-related predictors were among the most consistently selected environmental variables.

## Temporal Validation

Because the observations form a chronological time series, conventional random cross-validation was avoided.

Two temporal validation strategies were implemented.

### Sliding-window validation

Each fold used:

- 2 years for training;
- the following 2 years for validation.

A total of **10 temporal folds** were evaluated.

### Expanding-window validation

Training data progressively increased over time while the validation window remained chronological.

A total of **5 temporal folds** were evaluated.

The year **2023 was excluded from model development and reserved exclusively for final testing**.

## Models

Three machine-learning approaches were compared:

### LightGBM

Gradient-boosted decision trees designed for efficient learning on structured data.

### CatBoost

Gradient-boosting model with strong regularisation capabilities and robust performance on tabular datasets.

### Multilayer Perceptron

Feed-forward neural network used to investigate nonlinear relationships between environmental variables and the target.

Hyperparameters were selected separately for:

- each algorithm;
- each predictor subset;
- each temporal-validation strategy.

This produced **12 final model configurations**.

## Key Results

Performance on the independent 2023 test set showed **modest but consistent discriminatory ability**.

The best AUC-ROC was obtained by:

| Configuration | AUC-ROC |
|---|---:|
| CatBoost — Subset A — Sliding window | **0.624** |

One of the strongest sensitivity/specificity compromises was obtained with:

| Configuration | Sensitivity | Specificity |
|---|---:|---:|
| LightGBM — Subset B — Sliding window | 50.5% | 67.5% |

The MLP models generally prioritised sensitivity, detecting more positive days at the cost of a higher number of false positives.

Overall, the results indicate that environmental and temporal variables contain predictive information, but their predictive power alone is insufficient for reliable autonomous clinical forecasting.

Complete results are available in:

```text
reports/temporal_cv_results.csv
reports/final_test_metrics.csv
reports/shap_feature_importance.csv
```

## Explainability

SHAP was used to analyse how individual predictors influenced model predictions.

- **TreeExplainer** was used for LightGBM and CatBoost.
- **KernelExplainer** was used for the MLP.

NO₂-related variables showed some of the most consistent environmental contributions across models.

Seasonality also contributed substantially, particularly through the sinusoidal seasonal component.

SHAP values describe the behaviour of the trained models and **must not be interpreted as evidence of causal environmental effects on COPD exacerbations**.

## Repository Structure

```text
copd-exacerbation-forecasting/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_quality_and_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering_and_selection.ipynb
│   ├── 04_model_training_and_validation.ipynb
│   └── 05_final_evaluation_and_explainability.ipynb
│
├── reports/
│   ├── figures/
│   ├── feature_selection_results.json
│   ├── temporal_cv_results.csv
│   ├── final_test_metrics.csv
│   └── shap_feature_importance.csv
│
├── src/
│   └── copd_forecasting/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── selection.py
│       ├── validation.py
│       ├── models.py
│       ├── evaluation.py
│       └── explainability.py
│
├── tests/
│   ├── test_evaluation.py
│   ├── test_features.py
│   └── test_validation.py
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Notebook Workflow

The notebooks are designed to be followed sequentially:

### `01_data_quality_and_cleaning.ipynb`

- raw-data validation;
- temporal coverage checks;
- missing-data analysis;
- lag-boundary validation;
- anomalous-value handling;
- temporal interpolation.

### `02_exploratory_data_analysis.ipynb`

- target distribution;
- temporal patterns;
- population-group summaries;
- pollutant distributions;
- meteorological analysis;
- correlation analysis.

### `03_feature_engineering_and_selection.ipynb`

- temporal feature engineering;
- construction of predictor subsets A and B;
- correlation filtering;
- mRMR;
- temporal block bootstrap;
- final predictor selection.

### `04_model_training_and_validation.ipynb`

- LightGBM, CatBoost and MLP configuration;
- sliding-window temporal validation;
- expanding-window temporal validation;
- comparison of validation AUC.

### `05_final_evaluation_and_explainability.ipynb`

- training on the complete 2010–2022 development period;
- independent evaluation on 2023;
- sensitivity, specificity and predictive values;
- ROC and precision-recall metrics;
- confusion matrices;
- SHAP explainability.

## Installation

The project was developed using **Python 3.10**.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e 
```

## Tests

The repository includes automated tests for core components of the pipeline.

Run:

```bash
python -m pytest -v
```

The tests currently validate:

- binary-target construction;
- lag-variable naming;
- temporal-feature generation;
- expected dimensions of predictor subsets A and B;
- sliding-window temporal splits;
- expanding-window temporal splits;
- classification-metric calculations.

## Data Availability

The original dataset contains healthcare information and is therefore **not distributed with this repository**.

Raw and processed datasets are excluded through `.gitignore`.

The repository contains the analysis code, project structure and derived non-identifiable modelling results required to document the workflow without publishing the underlying clinical dataset.

Users with authorised access to the original data can place the daily dataset in the expected local data directory and reproduce the workflow sequentially.

## Methodological Note

This repository intentionally reproduces the modelling methodology used in the original Bachelor Thesis.

The independent **2023 test set remained completely isolated from model development**.

However, within the 2010–2022 development period, correlation filtering, scaling and feature selection were performed before internal temporal cross-validation.

A stricter prospective implementation would refit preprocessing and feature selection independently within each temporal training fold. Such a nested temporal pipeline represents a methodological extension beyond the original study and may produce different internal validation estimates.

## Limitations

The main limitations of the study include:

- retrospective and ecological design;
- aggregated rather than patient-level modelling;
- single geographical and healthcare setting;
- relatively low air-pollution levels;
- absence of potentially relevant clinical, infectious and behavioural predictors;
- effects associated with the COVID-19 period;
- correlated environmental predictors;
- absence of external validation.

The models should therefore be interpreted as a research exploration of environmental forecasting rather than as clinical decision-support systems.

## Tech Stack

**Data & scientific computing**

- Python
- pandas
- NumPy
- SciPy

**Machine learning**

- scikit-learn
- LightGBM
- CatBoost

**Explainable AI**

- SHAP

**Visualisation**

- Matplotlib
- Seaborn
- Plotly

**Development**

- Jupyter
- pytest
- Git

## Author

**Miguel Martín Nieto**

Bachelor's Degree in Biomedical Engineering  
University of Valladolid, Spain

Interests: Healthcare AI, clinical data science, machine learning and medical technology.