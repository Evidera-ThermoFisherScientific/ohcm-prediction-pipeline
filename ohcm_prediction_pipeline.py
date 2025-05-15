"""
OHCMPredictor: Pipeline for Predicting Obstructive Hypertrophic Cardiomyopathy (oHCM)
-------------------------------------------------------------------------------------
This script includes the full pipeline for:
1. Loading and transforming longitudinal patient data
2. Generating oHCM predictions using a trained XGBoost model
3. Profiling patients using a clustering model (KMeans on SHAP values)

Required Inputs:
- Preprocessed longitudinal patient-level data
- Trained XGBoost model (.pkl)
- Feature list used during model training
- Trained KMeans model for patient profiling

Dependencies:
- pandas==1.5.3
- numpy==1.23.5
- scipy==1.10.0
- scikit-learn==1.1.1
- xgboost==1.5.2
- shap==0.42.1
- Python==3.10.12
- s3fs==0.4.0 (if loading from S3)
"""

import os
import math
import pickle
import warnings
from typing import List

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
import xgboost as xgb

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)


def aggregate_features(df: pd.DataFrame,
                       df_out: pd.DataFrame,
                       lat_vals: List[str],
                       occur_features: List[str],
                       sum_features: List[str]) -> pd.DataFrame:
    """
    Aggregates longitudinal features for prediction.
    """
    for feature in lat_vals:
        recent = (
            df[['patid', 'monthnum', feature]]
            .dropna()
            .sort_values('monthnum')
            .groupby('patid')
            .last()
            .reset_index()
            .rename(columns={feature: f"{feature}_max"})
        )
        df_out = df_out.merge(recent, on='patid', how='left')

    for feature in occur_features:
        df_temp = df[['patid', 'monthnum', feature]].copy()
        df_temp[feature] = df_temp.apply(lambda x: x['monthnum'] if x[feature] > 0 else 0, axis=1)

        first_occur = (
            df_temp[df_temp[feature] > 0]
            .groupby('patid')[feature]
            .max()
            .rename(f'{feature}_first_occur')
            .reset_index()
        )

        recent_occur = (
            df_temp[df_temp[feature] > 0]
            .groupby('patid')[feature]
            .min()
            .rename(f'{feature}_recnt_occur')
            .reset_index()
        )

        merged = pd.merge(first_occur, recent_occur, on='patid', how='outer')
        df_out = df_out.merge(merged, on='patid', how='left')

    for feature in sum_features:
        summed = df.groupby('patid')[feature].sum(min_count=1).fillna(0).rename(f'{feature}_sum').reset_index()
        df_out = df_out.merge(summed, on='patid', how='left')

    return df_out


def recategorize_recent_occurrences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize recency of occurrences into ordinal bins.
    """
    for col in df.columns:
        if 'recnt_occur' in col:
            df[col] = df[col].apply(lambda x: 0 if x == 0 else (1 if x == 1 else (2 if 1 < x <= 12 else 3)))
    return df


def predict_ohcm(df: pd.DataFrame, model: xgb.Booster, features: List[str], threshold: float = 0.62) -> List[int]:
    """
    Predicts oHCM flag using a trained XGBoost model.
    """
    probs = model.predict_proba(df[features])
    return [1 if p[1] >= threshold else 0 for p in probs]


def generate_shap_profiles(model: xgb.Booster, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Generates SHAP value matrix for clustering.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data[features])
    return pd.DataFrame(shap_values, columns=features)


def assign_clusters(shap_df: pd.DataFrame, cluster_model: KMeans, mapping: dict) -> List[str]:
    """
    Assigns patient profile labels based on KMeans clusters.
    """
    labels = cluster_model.predict(shap_df)
    return [mapping.get(label, f"C{label}") for label in labels]


def run_pipeline(data_path: str,
                 model_path: str,
                 features_path: str,
                 kmeans_path: str,
                 output_path: str = "ohcm_results.csv") -> None:
    """
    Main function to run the oHCM prediction pipeline.
    """

    # Load data
    df_raw = pd.read_csv(data_path, sep=',')

    # Initialize target dataframe
    df_features = df_raw[['patid']].drop_duplicates()

    # Define variable categories
    lat_vals = ['value_smoke']
    occur_features = [
        'numevent_LFT', 'numdiag_CardiacDysrh', 'numdiag_Hyperten', 'numdiag_ChestPain',
        'numdiag_AnginaPect', 'numevent_UreaElec', 'numdiag_Breathless', 'numdiag_DiabType2',
        'numdiag_Hyperlip', 'numdiag_AtrialFib', 'numdiag_Palpitations', 'numdiag_oHCMfamily',
        'numdiag_Fatigue', 'numdiag_Anxiety', 'numpresc_Statins', 'numdiag_Fainting',
        'numevent_Biomarker', 'numdiag_COPD', 'numdiag_Hyperchol', 'numevent_CPET',
        'numpresc_Aspirin', 'numdiag_Thromb'
    ]
    sum_features = [
        'numevent_CPET', 'numevent_Biomarker', 'numevent_LFT', 'numevent_HospDay',
        'numevent_HospEmer', 'numevent_HospInpat', 'numevent_HospOut'
    ]

    # Aggregate features
    df_features = aggregate_features(df_raw, df_features, lat_vals, occur_features, sum_features)
    df_features = recategorize_recent_occurrences(df_features)

    # Load models and feature list
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(features_path, 'rb') as f:
        feature_list = pickle.load(f)

    with open(kmeans_path, 'rb') as f:
        kmeans = pickle.load(f)

    # Drop identifiers not used in modeling
    exclude_cols = ['patid', 'case_patid', 'indexdate']
    model_features = [f for f in feature_list if f not in exclude_cols]

    # Predict oHCM
    predictions = predict_ohcm(df_features, model, model_features)

    # Create result dataframe
    df_result = df_features[['patid']].copy()
    df_result['ohcm_flag'] = predictions

    # Compute SHAP values and assign patient profiles
    shap_df = generate_shap_profiles(model, df_features, model_features)

    cluster_label_map = {
        5: 'C0', 8: 'C1', 6: 'C2', 9: 'C3', 2: 'C4',
        3: 'C5', 0: 'C6', 4: 'C7', 7: 'C8', 1: 'C9'
    }

    df_result['patient_profile'] = assign_clusters(shap_df, kmeans, cluster_label_map)

    # Save result
    df_result.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Example run (paths should be updated accordingly)
    run_pipeline(
        data_path='path_to_input_file.csv',
        model_path='model/ohcm_ml_model.pkl',
        features_path='model/features.pkl',
        kmeans_path='model/kmeans_model.pkl',
        output_path='results/ohcm_results.csv'
    )
