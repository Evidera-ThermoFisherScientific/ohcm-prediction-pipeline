# OHCMPredictor: Pipeline for Predicting Obstructive Hypertrophic Cardiomyopathy (oHCM)

This repository contains the full machine learning pipeline for predicting oHCM and profiling patients based on longitudinal data and SHAP values using XGBoost and KMeans.

## 🚀 Features

- Aggregates and transforms patient-level data
- Predicts oHCM status using a trained XGBoost model
- Generates SHAP value profiles for interpretability
- Clusters patients into meaningful profiles using KMeans


## 📦 Requirements

```bash
pip install -r requirements.txt



## 📊 Input Data Requirements

The model expects **60 months of longitudinal data** (monthly granularity) for each patient. Your input CSV should include the following:

- `patid`: Unique patient ID
- `monthnum`: Month index (e.g., 1–60)
- Longitudinal features:
  - Smoking status (`value_smoke`)
  - Event-based features (e.g., `numevent_CPET`, `numevent_LFT`, etc.)
  - Diagnosis-based features (e.g., `numdiag_Hyperten`, `numdiag_AtrialFib`, etc.)
  - Prescription-based features (e.g., `numpresc_Statins`, `numpresc_Aspirin`, etc.)

Refer to [`data_dictionary/ohcm_data_dictionary.xlsx`](./data_dictionary/ohcm_data_dictionary.xlsx) for a detailed description of each feature.

---

## 🚀 How to Run

modify the __main__ block as follows:

run_pipeline(
    data_path='data/longitudinal_input.csv',
    model_path='model/ohcm_ml_model.pkl',
    features_path='model/features.pkl',
    kmeans_path='model/kmeans_model.pkl',
    output_path='results/ohcm_results.csv'
)

and run

python ohcm_prediction_pipeline.py


## 📈 Output

The output will be a CSV file (ohcm_results.csv) with:

patid: Patient identifier

ohcm_flag: Model prediction (1 = oHCM likely, 0 = not likely)

patient_profile: Cluster label based on SHAP value profiling (e.g., C0, C1, ..., C9). C0 - C3 ohcm high risk patients
