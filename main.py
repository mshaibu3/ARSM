from src.wearable_integration.data_collector import collect_data_from_wearable, send_data_to_cloud
from src.analytics.predictive_model import train_model, predict
from src.utils.helpers import load_and_preprocess_data
from src.analytics.multi_sensor_fusion import fuse_sensor_data
from src.analytics.anomaly_detection import detect_anomalies
from src.cloud.cloud_integration import upload_to_s3, download_from_s3

# Example API URLs
wearable_api_url = "http://example.com/wearable_data"
cloud_url = "http://localhost:5000/data"
s3_bucket_name = "your-s3-bucket"
s3_object_name = "data.csv"
local_file_path = "path/to/your/dataset.csv"

# Collect data from wearable
data = collect_data_from_wearable(wearable_api_url)

# Send data to cloud
status_code = send_data_to_cloud(data, cloud_url)
print(f"Data sent to cloud with status code: {status_code}")

# Load and preprocess data
data = load_and_preprocess_data(local_file_path)

# Fuse sensor data
ecg_data = data[:, :100]  # Example slicing
ppg_data = data[:, 100:200]
spo2_data = data[:, 200:300]
motion_data = data[:, 300:]
fused_data = fuse_sensor_data(ecg_data, ppg_data, spo2_data, motion_data)

# Detect anomalies
anomalies = detect_anomalies(fused_data)
print(f"Anomalies detected: {anomalies}")

# Train model
X, y = fused_data[:, :-1], fused_data[:, -1]
model = train_model(X, y)

# Predict risk
risk = predict(model, X)
print(f"Predicted risk: {risk}")

# Upload data to S3
upload_to_s3(local_file_path, s3_bucket_name, s3_object_name)

# Download data from S3
download_from_s3(s3_bucket_name, s3_object_name, local_file_path)
