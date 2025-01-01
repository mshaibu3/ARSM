import numpy as np

def fuse_sensor_data(ecg_data, ppg_data, spo2_data, motion_data):
    """
    Fuse data from multiple sensors.
    """
    # Assuming all data arrays have the same length
    fused_data = np.hstack((ecg_data, ppg_data, spo2_data, motion_data))
    return fused_data
