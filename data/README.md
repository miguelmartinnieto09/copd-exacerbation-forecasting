# Data

This directory is reserved for the datasets used in the COPD exacerbation forecasting project.

The original healthcare dataset is **not included in this repository** because it contains clinical information that cannot be publicly distributed.

## Expected Structure

```text
data/
├── README.md
├── raw/
│   └── copd_daily_data.xlsx
└── processed/
    └── copd_daily_data_cleaned.xlsx
```

## Raw Data

The analysis uses an aggregated daily dataset containing COPD-related emergency department activity together with environmental and meteorological variables.

The expected input file is:

```text
data/raw/copd_daily_data.xlsx
```

The modelling workflow uses the daily aggregated worksheet from the original study dataset.

The study period covers:

```text
2010-01-01 → 2023-12-31
```

The raw dataset includes:

- daily COPD-related emergency department counts;
- air-pollution measurements;
- meteorological variables;
- environmental variables with temporal lags;
- calendar information used for subsequent feature engineering.

## Processed Data

Notebook:

```text
notebooks/01_data_quality_and_cleaning.ipynb
```

performs the data-quality and cleaning workflow and generates:

```text
data/processed/copd_daily_data_cleaned.xlsx
```

The preprocessing workflow includes:

- study-period filtering;
- chronological consistency checks;
- validation of lag-related missingness;
- replacement of invalid measurements;
- removal of the initial lag-boundary observations;
- interpolation of missing environmental observations.

The processed dataset is subsequently used by the exploratory analysis, feature-selection and modelling notebooks.

## Privacy and Data Availability

The original data were obtained for an academic biomedical research project and include healthcare-related information.

For this reason:

- raw data are not publicly distributed;
- processed data are not publicly distributed;
- both `data/raw/` and `data/processed/` are excluded from version control;
- only non-identifiable derived modelling results are stored in the public repository.

The repository therefore contains the complete analysis pipeline without exposing the underlying clinical dataset.

Users with authorised access to the original data can place the required file in:

```text
data/raw/copd_daily_data.xlsx
```

and execute the notebooks sequentially to reproduce the workflow.