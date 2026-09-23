# Water Intelligence — Environmental Machine Learning Platform

## Overview

Water Intelligence is an end-to-end Machine Learning project designed to analyze
and model different environmental and water-related factors.

The project combines:

- Water consumption forecasting
- Flood risk prediction
- Rainfall analysis
- Soil moisture analysis
- Environmental clustering
- Environmental anomaly detection

The project follows a complete Machine Learning workflow:

Data Collection → Data Cleaning → Exploratory Data Analysis →
Feature Engineering → Model Training → Model Evaluation →
Unsupervised Learning → Model Saving → Reporting

---

## Objectives

The main objectives of the project are:

1. Forecast water consumption using historical consumption patterns.
2. Predict flood occurrence using environmental features.
3. Analyze rainfall patterns across years and subdivisions.
4. Analyze soil moisture trends across locations and dates.
5. Identify environmental patterns using clustering.
6. Detect unusual environmental observations using anomaly detection.
7. Build reusable Machine Learning models and save them for future use.

---

## Datasets

The project uses four datasets.

### 1. Water Consumption Dataset

File:

`water_consumption_forecasting.csv`

Used for:

- Water consumption forecasting
- Time-series feature engineering
- Regional consumption analysis

Important features include:

- Date
- Region
- Consumption in liters

---

### 2. Flood Risk Dataset

File:

`flood_risk_dataset_india.csv`

Used for:

- Flood occurrence classification
- Environmental risk analysis
- Clustering
- Anomaly detection

The target variable is:

`Flood Occurred`

---

### 3. Rainfall Dataset

File:

`Sub_Division_IMD_2017.csv`

Used for:

- Annual rainfall analysis
- Monthly rainfall analysis
- Rainfall trend identification
- Rainfall feature engineering

Important information includes:

- Year
- Subdivision
- Monthly rainfall
- Annual rainfall

---

### 4. Soil Moisture Dataset

File:

`sm_telanganа_2020.csv`

Used for:

- Soil moisture trend analysis
- District-wise analysis
- Temporal feature engineering
- Soil moisture rolling averages

---

# Project Architecture

```text
water-intelligence/
│
├── datasets/
│   ├── water_consumption_forecasting.csv
│   ├── flood_risk_dataset_india.csv
│   ├── Sub_Division_IMD_2017.csv
│   └── sm_telanganа_2020.csv
│
├── data/
│   ├── cleaned/
│   │   ├── water_consumption_cleaned.csv
│   │   ├── flood_risk_cleaned.csv
│   │   ├── rainfall_cleaned.csv
│   │   └── soil_moisture_cleaned.csv
│   │
│   └── features/
│       ├── water_features.csv
│       ├── flood_features.csv
│       ├── rainfall_features.csv
│       └── soil_moisture_features.csv
│
├── notebooks/
│   └── Water_Intelligence_ML.ipynb
│
├── models/
│   ├── Water consumption models
│   ├── Flood risk models
│   ├── K-Means model
│   ├── PCA model
│   ├── Isolation Forest model
│   └── Cluster scaler
│
├── reports/
│   ├── water_model_results.csv
│   ├── flood_model_results.csv
│   ├── environmental_clusters.csv
│   ├── cluster_profiles.csv
│   └── environmental_anomalies.csv
│
└── README.md