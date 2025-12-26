"""
Test suite for Phishing Detection API.
Demonstrates all endpoints and expected responses.
"""

from api_client import PhishingDetectionClient
import json

def test_email_detection():
    """Test email phishing detection"""
    client = PhishingDetectionClient()
    
    # Test case 1: Legitimate email
    legitimate_email = """
    Hi John,
    
    Hope you're doing well. Here's the quarterly report you requested.
    
    Best regards,
    Sarah
    """
    
    result = client.detect_email(
        email_body=legitimate_email,
        sender='sarah@company.com'
    )
    print("Legitimate Email Result:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*60 + "\n")
    
    # Test case 2: Phishing email
    phishing_email = """
    Dear User,
    
    Unusual activity detected on your account. Please click here immediately 
    to verify your identity and update your banking details.
    
    https://verify-account-security.suspicious-domain.tk/login
    
    This is urgent! Act now or your account will be closed.
    
    Regards,
    Security Team
    """
    
    result = client.detect_email(
        email_body=phishing_email,
        sender='security@verify-account.tk'
    )
    print("Phishing Email Result:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*60 + "\n")

def test_url_detection():
    """Test URL phishing detection"""
    client = PhishingDetectionClient()
    
    # Test case 1: Legitimate URL
    result = client.detect_url('https://github.com/user/repo')
    print("Legitimate URL Result:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*60 + "\n")
    
    # Test case 2: Phishing URL
    result = client.detect_url('https://googlе-security-verify.tk/redirect?go=phishing-site')
    print("Phishing URL Result:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*60 + "\n")

def test_sms_detection():
    """Test SMS phishing detection"""
    client = PhishingDetectionClient()
    
    # Test case 1: Legitimate SMS
    result = client.detect_sms('Hi Sarah, just confirming our meeting tomorrow at 2pm. -John')
    print("Legitimate SMS Result:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*60 + "\n")
    
    # Test case 2: Smishing
    result = client.detect_sms('URGENT: Verify your PayPal account immediately! bit.ly/secure-paypal')
    print("Smishing Result:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*60 + "\n")

def test_statistics():
    """Test statistics endpoint"""
    client = PhishingDetectionClient()
    
    result = client.get_statistics()
    print("Statistics Result:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*60 + "\n")

def test_detection_history():
    """Test detection history endpoint"""
    client = PhishingDetectionClient()
    
    result = client.get_detection_history(limit=10, classification='malicious')
    print("Malicious Detections History:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*60 + "\n")

def test_whitelist_blacklist():
    """Test whitelist/blacklist operations"""
    client = PhishingDetectionClient()
    
    # Add to whitelist
    result = client.add_whitelist(sender='trusted@company.com')
    print("Whitelist Add Result:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*60 + "\n")
    
    # Add to blacklist
    result = client.add_blacklist(domain='phishing-domain.tk')
    print("Blacklist Add Result:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    print("PHISHING DETECTION API TEST SUITE")
    print("="*60 + "\n")
    
    test_email_detection()
    test_url_detection()
    test_sms_detection()
    test_statistics()
    test_detection_history()
    test_whitelist_blacklist()
    
    print("All tests completed!")
