from flask import Flask, request, jsonify, render_template
import firebase_admin
from firebase_admin import credentials, firestore
from src.analytics.predictive_model import build_cnn_lstm_model, predict
from src.utils.helpers import preprocess_data
import numpy as np

app = Flask(__name__)

cred = credentials.Certificate("path/to/your/firebase/credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['POST'])
def receive_data():
    """
    Receive data from wearable devices and store it in Firestore.
    """
    data = request.json
    db.collection('health_data').add(data)
    return jsonify({"status": "success"}), 200

@app.route('/risk', methods=['GET'])
def get_risk():
    """
    Fetch data from Firestore, preprocess it, and predict risk levels.
    """
    docs = db.collection('health_data').stream()
    data = [doc.to_dict() for doc in docs]
    
    data = preprocess_data(data)
    
    model = build_cnn_lstm_model((data.shape[1], data.shape[2]))
    risk_level = predict(model, data)
    
    return jsonify({"risk_level": risk_level.tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True)
