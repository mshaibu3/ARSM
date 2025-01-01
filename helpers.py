import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def handle_missing_values(data):
    """
    Handle missing values using KNN imputation.
    """
    imputer = KNNImputer(n_neighbors=5)
    return imputer.fit_transform(data)

def normalize_features(data):
    """
    Normalize features using StandardScaler.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def preprocess_data(data):
    """
    Preprocess data by handling missing values and normalizing features.
    """
    data = handle_missing_values(data)
    data = normalize_features(data)
    return data

def load_and_preprocess_data(file_path):
    """
    Load and preprocess data from a CSV file.
    """
    data = pd.read_csv(file_path)
    data = preprocess_data(data)
    return data
