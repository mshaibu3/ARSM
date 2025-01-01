import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(data):
    """
    Detect anomalies in the data using Isolation Forest.
    """
    model = IsolationForest(contamination=0.01)
    model.fit(data)
    anomalies = model.predict(data)
    return anomalies
