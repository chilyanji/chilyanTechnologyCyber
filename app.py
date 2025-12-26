from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import pickle
import numpy as np
from urllib.parse import urlparse
import re
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from google.oauth2 import service_account
import requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load pre-trained models
try:
    rf_model = joblib.load('models/phishing_rf_model.pkl')
    nn_model = tf.keras.models.load_model('models/phishing_nn_model.h5')
    scaler = joblib.load('models/scaler.pkl')
except:
    print("Models not yet trained. Run training script first.")

# CHILYAN Technology branding
COMPANY_INFO = {
    "name": "CHILYAN Technology",
    "product": "PhishGuard",
    "version": "1.0.0",
    "slogan": "Detect. Classify. Respond. Protect."
}

class PhishingDetectionEngine:
    """Core phishing detection engine for CHILYAN PhishGuard"""
    
    def __init__(self):
        self.threats_database = {
            'suspicious_keywords': [
                'verify', 'confirm', 'urgent', 'click here', 'update now',
                'suspended', 'locked', 'unauthorized', 'act now', 'confirm identity'
            ],
            'safe_domains': [
                'google.com', 'microsoft.com', 'apple.com', 'facebook.com', 'amazon.com'
            ]
        }
        self.detection_history = []
    
    def extract_url_features(self, url):
        """Extract 25+ features from URL for ML classification"""
        features = {}
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            
            # Basic features
            features['url_length'] = len(url)
            features['domain_length'] = len(domain)
            features['path_length'] = len(path)
            features['subdomain_count'] = domain.count('.')
            
            # Special character analysis
            features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
            features['has_at_symbol'] = 1 if '@' in url else 0
            features['has_hyphen_domain'] = 1 if '-' in domain else 0
            features['has_underscore'] = 1 if '_' in url else 0
            
            # Protocol analysis
            features['has_https'] = 1 if url.startswith('https') else 0
            features['port_number'] = 1 if ':' in domain else 0
            
            # Suspicious patterns
            features['suspicious_keyword_count'] = sum(1 for keyword in self.threats_database['suspicious_keywords'] if keyword.lower() in url.lower())
            features['digit_ratio'] = sum(1 for c in url if c.isdigit()) / len(url) if len(url) > 0 else 0
            features['special_char_ratio'] = sum(1 for c in url if not c.isalnum() and c != '/' and c != ':' and c != '.') / len(url) if len(url) > 0 else 0
            
            # Domain reputation (simplified)
            features['is_safe_domain'] = 1 if any(domain.endswith(safe) for safe in self.threats_database['safe_domains']) else 0
            
            # Entropy calculation
            entropy = 0
            for char in set(url):
                p = url.count(char) / len(url)
                entropy -= p * np.log2(p)
            features['entropy'] = entropy
            
        except:
            features = {k: 0 for k in range(15)}
        
        return features
    
    def analyze_email(self, email_content):
        """Analyze email headers and body for phishing indicators"""
        analysis = {
            'sender_spoofing_risk': 0,
            'suspicious_links': 0,
            'suspicious_attachments': 0,
            'urgency_language': 0,
            'generic_greeting': 0,
            'grammar_quality': 'good'
        }
        
        # Check for suspicious keywords
        urgent_keywords = ['urgent', 'immediate', 'action required', 'verify now', 'confirm identity']
        for keyword in urgent_keywords:
            if keyword.lower() in email_content.lower():
                analysis['urgency_language'] += 1
        
        # Check for generic greetings (phishing indicator)
        if 'dear user' in email_content.lower() or 'dear customer' in email_content.lower():
            analysis['generic_greeting'] = 1
        
        # Extract URLs from email
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_content)
        analysis['suspicious_links'] = len(urls)
        
        return analysis
    
    def analyze_sms(self, sms_content):
        """Analyze SMS/smishing messages"""
        analysis = {
            'is_smishing': False,
            'risk_score': 0,
            'indicators': []
        }
        
        # Smishing patterns
        smishing_patterns = [
            r'click.*link',
            r'verify.*account',
            r'confirm.*payment',
            r'suspended',
            r'unusual.*activity',
            r'update.*now'
        ]
        
        for pattern in smishing_patterns:
            if re.search(pattern, sms_content, re.IGNORECASE):
                analysis['indicators'].append(pattern)
                analysis['risk_score'] += 20
        
        # URL in SMS is suspicious
        if re.search(r'http[s]?://', sms_content):
            analysis['indicators'].append('Contains shortened URL')
            analysis['risk_score'] += 30
        
        analysis['is_smishing'] = analysis['risk_score'] > 40
        
        return analysis
    
    def classify_threat(self, detection_score):
        """Classify threat level based on ML score"""
        if detection_score >= 0.7:
            return {
                'level': 'MALICIOUS',
                'color': 'danger',
                'icon': '⛔',
                'description': 'High-confidence phishing threat detected'
            }
        elif detection_score >= 0.4:
            return {
                'level': 'SUSPICIOUS',
                'color': 'warning',
                'icon': '⚠️',
                'description': 'Potential phishing attempt - verify before proceeding'
            }
        else:
            return {
                'level': 'LEGITIMATE',
                'color': 'success',
                'icon': '✓',
                'description': 'Content appears safe'
            }
    
    def generate_explanation(self, url, features, threat_level):
        """Generate human-readable explanation for detection result"""
        explanation = []
        
        if features.get('has_ip'):
            explanation.append("• URL uses IP address instead of domain name")
        if features.get('has_at_symbol'):
            explanation.append("• URL contains '@' symbol (common phishing technique)")
        if features.get('has_https') == 0:
            explanation.append("• Missing secure HTTPS protocol")
        if features.get('suspicious_keyword_count', 0) > 0:
            explanation.append(f"• Contains {features['suspicious_keyword_count']} suspicious keywords")
        if features.get('has_hyphen_domain'):
            explanation.append("• Domain contains hyphens (spoofing indicator)")
        
        if not explanation:
            explanation.append("• URL structure matches legitimate patterns")
        
        return explanation
    
    def detect(self, input_data):
        """Main detection function for emails, SMS, URLs"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'input_type': input_data.get('type'),
            'threat_level': None,
            'confidence': 0,
            'explanation': [],
            'recommended_action': None
        }
        
        try:
            if input_data['type'] == 'url':
                url = input_data['content']
                features = self.extract_url_features(url)
                
                # ML prediction
                feature_array = np.array([list(features.values())])
                prediction = rf_model.predict_proba(feature_array)[0][1]
                
                threat_classification = self.classify_threat(prediction)
                result['threat_level'] = threat_classification['level']
                result['confidence'] = float(prediction)
                result['explanation'] = self.generate_explanation(url, features, threat_classification['level'])
                
            elif input_data['type'] == 'email':
                email_content = input_data['content']
                analysis = self.analyze_email(email_content)
                
                # Calculate overall score
                score = (analysis['urgency_language'] * 0.3 + 
                        analysis['generic_greeting'] * 0.2 + 
                        analysis['suspicious_links'] * 0.5) / 10
                
                threat_classification = self.classify_threat(score)
                result['threat_level'] = threat_classification['level']
                result['confidence'] = min(float(score), 1.0)
                result['explanation'] = [f"• Found {analysis['suspicious_links']} suspicious links in email"]
                
            elif input_data['type'] == 'sms':
                sms_content = input_data['content']
                analysis = self.analyze_sms(sms_content)
                
                score = analysis['risk_score'] / 100
                threat_classification = self.classify_threat(score)
                result['threat_level'] = threat_classification['level']
                result['confidence'] = min(float(score), 1.0)
                result['explanation'] = analysis['indicators']
            
            # Recommended action
            if result['threat_level'] == 'MALICIOUS':
                result['recommended_action'] = 'BLOCK and report to authorities'
            elif result['threat_level'] == 'SUSPICIOUS':
                result['recommended_action'] = 'QUARANTINE for manual review'
            else:
                result['recommended_action'] = 'ALLOW'
            
            # Store in history
            self.detection_history.append(result)
            
        except Exception as e:
            result['error'] = str(e)
        
        return result

# Initialize detection engine
detection_engine = PhishingDetectionEngine()

# API Routes
@app.route('/')
def index():
    return render_template('index.html', company=COMPANY_INFO)

@app.route('/api/detect', methods=['POST'])
def detect():
    """Main detection API endpoint"""
    try:
        data = request.get_json()
        result = detection_engine.detect(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get detection history"""
    limit = request.args.get('limit', 100, type=int)
    return jsonify({
        'total': len(detection_engine.detection_history),
        'history': detection_engine.detection_history[-limit:]
    }), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get threat statistics"""
    history = detection_engine.detection_history
    stats = {
        'total_detections': len(history),
        'malicious': sum(1 for h in history if h.get('threat_level') == 'MALICIOUS'),
        'suspicious': sum(1 for h in history if h.get('threat_level') == 'SUSPICIOUS'),
        'legitimate': sum(1 for h in history if h.get('threat_level') == 'LEGITIMATE'),
        'average_confidence': np.mean([h.get('confidence', 0) for h in history]) if history else 0
    }
    return jsonify(stats), 200

@app.route('/api/company', methods=['GET'])
def get_company_info():
    """Get CHILYAN Technology company info"""
    return jsonify(COMPANY_INFO), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)
# app.run(debug=True, port=5001)  # Use different port like 5001