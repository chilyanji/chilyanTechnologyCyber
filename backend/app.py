from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
from detection_engine import PhishingDetectionEngine
from database import Database

app = Flask(__name__)
CORS(app)

# Initialize detection engine and database
detector = PhishingDetectionEngine()
db = Database()

@app.route('/api/detect', methods=['POST'])
def detect_threat():
    """
    Main endpoint for phishing detection.
    Accepts: email body, sender, URL, SMS text
    Returns: Classification, confidence score, explanation
    """
    try:
        data = request.json
        
        # Validate input
        if not data or not any(k in data for k in ['email_body', 'url', 'sms_text']):
            return jsonify({'error': 'Please provide email_body, url, or sms_text'}), 400
        
        # Extract fields
        email_body = data.get('email_body', '')
        sender = data.get('sender', '')
        url = data.get('url', '')
        sms_text = data.get('sms_text', '')
        
        # Run detection
        result = detector.analyze(
            email_body=email_body,
            sender=sender,
            url=url,
            sms_text=sms_text
        )
        
        # Store in database
        detection_record = {
            'timestamp': datetime.now().isoformat(),
            'input_type': data.get('type', 'email'),
            'classification': result['classification'],
            'confidence': result['confidence'],
            'features': result['features'],
            'explanation': result['explanation']
        }
        
        db.store_detection(detection_record)
        
        # Trigger response if malicious
        if result['classification'] == 'malicious':
            trigger_response(result, data)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Retrieve detection history with optional filters"""
    limit = request.args.get('limit', 100, type=int)
    classification = request.args.get('classification', None)
    
    history = db.get_detections(limit=limit, classification=classification)
    return jsonify({'detections': history}), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get detection statistics"""
    stats = db.get_stats()
    return jsonify(stats), 200

@app.route('/api/whitelist', methods=['POST'])
def add_whitelist():
    """Add sender or domain to whitelist"""
    data = request.json
    db.add_whitelist(data.get('sender'), data.get('domain'))
    return jsonify({'message': 'Added to whitelist'}), 200

@app.route('/api/blacklist', methods=['POST'])
def add_blacklist():
    """Add sender or domain to blacklist"""
    data = request.json
    db.add_blacklist(data.get('sender'), data.get('domain'))
    return jsonify({'message': 'Added to blacklist'}), 200

def trigger_response(result, data):
    """Trigger automated response for malicious threats"""
    response_action = {
        'timestamp': datetime.now().isoformat(),
        'threat_classification': result['classification'],
        'actions': [
            'quarantine',  # Quarantine the message
            'alert',       # Send security alert
            'log'          # Log for audit trail
        ],
        'target': data.get('recipient', 'unknown')
    }
    db.store_response_log(response_action)
    # In production: send alerts to security team, block sender, etc.

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
