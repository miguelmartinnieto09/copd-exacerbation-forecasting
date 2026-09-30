# Bachelor Thesis: Machine Learning for COPD Exacerbation Forecasting

Machine learning project developed as my Bachelor's Thesis in Biomedical Engineering at the University of Valladolid.

The project investigates whether **air pollution, meteorological and temporal variables can help predict the daily occurrence of emergency department visits due to Chronic Obstructive Pulmonary Disease (COPD) exacerbations**.

## Overview

COPD exacerbations are clinically relevant events that frequently lead to emergency department visits and hospital admissions.

Although their causes are multifactorial, environmental conditions may act as triggers or modulators. This project explored whether these factors contain enough predictive information to distinguish between days with and without COPD-related emergency department visits.

The study used aggregated daily data from the Valladolid Oeste Health Area, Spain, covering **2010–2023**.

## Problem Definition

The original outcome represented the daily number of COPD-related emergency department visits.

For modelling, it was transformed into a binary classification target:

- **0:** no COPD-related emergency department visit.
- **1:** at least one COPD-related emergency department visit.

Healthcare records were combined with:

- Air-pollution variables.
- Meteorological variables.
- Temporal and seasonal information.

Data from **2023 were reserved as an independent final test period**, while previous years were used for model development and temporal validation.

## Machine Learning Pipeline

The workflow included:

### 1. Data curation and integration
Integration and preprocessing of healthcare, environmental, meteorological and temporal data.

### 2. Feature engineering
Two representations of environmental exposure were evaluated:

- Immediate and accumulated variables.
- Individual temporal lags from 0 to 7 days.

### 3. Feature selection
Feature selection was performed using:

- Correlation filtering.
- Minimum Redundancy Maximum Relevance (mRMR).
- Temporal block bootstrap to evaluate feature-selection stability.

### 4. Predictive modelling
Three classification approaches were developed and compared:

- **LightGBM**
- **CatBoost**
- **Multilayer Perceptron (MLP)**

### 5. Temporal validation
Models were optimized using two time-aware validation strategies:

- Sliding-window validation.
- Expanding-window validation.

### 6. Final evaluation
Final performance was assessed independently using the **2023 test period**.

### 7. Explainable AI
Model behaviour and feature contributions were analysed using **SHAP**.

## Key Results

The models showed **modest discriminative performance**, indicating that environmental and temporal variables contained useful predictive information but were not sufficient on their own for reliable clinical prediction.

- **CatBoost** achieved the highest AUC-ROC: **0.624**.
- **LightGBM** achieved a sensitivity of **50.5%** and specificity of **67.5%** in one evaluated configuration.
- **MLP** achieved the highest sensitivity: **69.8%**, with a lower specificity of **39.9%**.

No model was consistently superior across all evaluation metrics.

### Feature interpretation

**NO₂ was the most consistent environmental predictor** across the analyses.

Higher NO₂ concentrations, both immediate values and some delayed exposures, tended to contribute positively to the predicted probability of COPD-related emergency department visits.

Seasonality also provided relevant information and reflected the higher frequency of exacerbations during colder periods.

Other variables, including particulate matter, atmospheric pressure, precipitation and ozone, contributed in some configurations but showed less stable patterns.

## Interpretation and Limitations

The results suggest that environmental and temporal information contributes to the prediction of COPD-related healthcare demand, but the resulting models were **not sufficiently reliable for autonomous clinical use**.

Several factors limit the interpretation and generalisation of the results:

- The study was **retrospective, ecological and single-centre**.
- Valladolid represents a **relatively low-pollution urban environment**, limiting environmental variability.
- Air-pollution measurements represented population-level exposure rather than individual exposure.
- The daily number of COPD-related emergency visits was relatively small.
- Important determinants such as respiratory infections, previous exacerbations, smoking and individual clinical characteristics were not available.
- The study period included major changes in healthcare behaviour and environmental conditions, particularly during the COVID-19 pandemic.
- Strong correlations existed between pollutants, meteorological variables and temporal lags.
- Feature importance and SHAP values describe predictive behaviour and should **not be interpreted as evidence of causal effects**.
- No external validation cohort from another hospital or city was available.

The results should therefore be interpreted as evidence of **predictive signal under the studied conditions**, rather than as a clinically deployable prediction system.

## Future Work

Future development could include:

- External validation using additional hospitals, healthcare areas or cities.
- Integration of clinical, demographic, infectious and healthcare-related variables.
- Alternative definitions of the prediction target.
- Count-based modelling of emergency department demand.
- Prediction of high-demand days.
- Evaluation in environments with different pollution and climate profiles.

## Data Availability

The original healthcare data used in this study are **not included in this repository**.

The repository provides the code, methodology and machine learning workflow while respecting the confidentiality and usage restrictions associated with the original healthcare data.

## Repository Structure

```text
copd-exacerbation-forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── evaluation/
│
├── reports/
│   └── figures/
│
└── tests/
```

## Author

**Miguel Martín Nieto**  
Biomedical Engineer  
Machine Learning · Healthcare Data · Medical AI