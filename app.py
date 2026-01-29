"""
Flask Backend Server for Dengue Prediction
Serves the ML model via REST API
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import sys
from ml_model import load_model, predict_dengue

# ========================
# FLASK APP SETUP
# ========================
app = Flask(__name__)
CORS(app)

# Load ML model at startup
print("🚀 Loading ML model...")
model, feature_names = load_model()

if model is None or feature_names is None:
    print("❌ Failed to load ML model. Server may not work properly.")
else:
    print(f"✅ ML model loaded successfully!")
    print(f"📊 Model uses {len(feature_names)} features")

# ========================
# ROUTES
# ========================

@app.route('/')
def home():
    """Server health check"""
    return jsonify({
        'status': 'running',
        'message': 'Dengue ML Prediction Server Active',
        'model_loaded': model is not None
    })

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model': 'loaded' if model is not None else 'not_loaded',
        'features': len(feature_names) if feature_names else 0
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict dengue from blood sample data
    
    Expected JSON:
    {
        "ns1": value,
        "igg": value,
        "igm": value,
        "hemoglobin": value,
        "platelet_count": value,
        ... other features
    }
    """
    
    if model is None or feature_names is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        result = predict_dengue(data, feature_names, model)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict dengue for multiple blood samples
    
    Expected JSON:
    {
        "samples": [
            {...sample1...},
            {...sample2...},
            ...
        ]
    }
    """
    
    if model is None or feature_names is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        samples = data.get('samples', [])
        
        if not samples:
            return jsonify({'error': 'No samples provided'}), 400
        
        results = []
        for sample in samples:
            result = predict_dengue(sample, feature_names, model)
            results.append(result)
        
        return jsonify({
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Batch prediction failed'
        }), 500

@app.route('/api/features')
def get_features():
    """Get list of required features"""
    return jsonify({
        'features': feature_names if feature_names else [],
        'count': len(feature_names) if feature_names else 0
    })

# ========================
# ERROR HANDLERS
# ========================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ========================
# RUN SERVER
# ========================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 DENGUE ML PREDICTION SERVER")
    print("=" * 60)
    print("📡 Server running on http://localhost:5000")
    print("🔗 API endpoints:")
    print("   - GET  /                 - Health check")
    print("   - GET  /api/health       - Detailed health check")
    print("   - POST /api/predict      - Single prediction")
    print("   - POST /api/predict-batch- Multiple predictions")
    print("   - GET  /api/features     - List features")
    print("=" * 60)
    
    app.run(host='localhost', port=5000, debug=False)
