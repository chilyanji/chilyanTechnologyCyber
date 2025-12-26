import requests
import json
from typing import Dict, Optional

class PhishingDetectionClient:
    """
    Client for interacting with the Phishing Detection API.
    Provides methods for all detection and management operations.
    """
    
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
    
    def detect_email(self, email_body: str, sender: str = '') -> Dict:
        """
        Detect phishing in email.
        
        Args:
            email_body: Full email content
            sender: Email sender address
        
        Returns:
            Detection result with classification and explanation
        """
        payload = {
            'type': 'email',
            'email_body': email_body,
            'sender': sender
        }
        return self._make_request('/api/detect', payload)
    
    def detect_url(self, url: str) -> Dict:
        """
        Detect phishing in URL.
        
        Args:
            url: URL to analyze
        
        Returns:
            Detection result
        """
        payload = {'type': 'url', 'url': url}
        return self._make_request('/api/detect', payload)
    
    def detect_sms(self, sms_text: str) -> Dict:
        """
        Detect smishing (SMS phishing).
        
        Args:
            sms_text: SMS message content
        
        Returns:
            Detection result
        """
        payload = {'type': 'sms', 'sms_text': sms_text}
        return self._make_request('/api/detect', payload)
    
    def get_detection_history(self, limit: int = 100, classification: Optional[str] = None) -> Dict:
        """Get detection history"""
        params = {'limit': limit}
        if classification:
            params['classification'] = classification
        
        return self._make_request('/api/history', params=params)
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return self._make_request('/api/stats')
    
    def add_whitelist(self, sender: str = '', domain: str = '') -> Dict:
        """Add sender or domain to whitelist"""
        payload = {'sender': sender, 'domain': domain}
        return self._make_request('/api/whitelist', payload)
    
    def add_blacklist(self, sender: str = '', domain: str = '') -> Dict:
        """Add sender or domain to blacklist"""
        payload = {'sender': sender, 'domain': domain}
        return self._make_request('/api/blacklist', payload)
    
    def _make_request(self, endpoint: str, payload: Dict = None, params: Dict = None) -> Dict:
        """Make HTTP request to API"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if payload:
                response = requests.post(url, json=payload)
            else:
                response = requests.get(url, params=params)
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}
