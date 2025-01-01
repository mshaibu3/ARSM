import requests
import json

def collect_data_from_wearable(api_url):
    """
    Collect data from a wearable device.
    """
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def send_data_to_cloud(data, cloud_url):
    """
    Send data to the cloud.
    """
    headers = {'Content-Type': 'application/json'}
    response = requests.post(cloud_url, data=json.dumps(data), headers=headers)
    return response.status_code
