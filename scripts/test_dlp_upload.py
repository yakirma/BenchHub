import requests
import base64
import sys

def test_dlp_safe_upload():
    url = "http://localhost:6060/metrics/create"
    
    # Sample Python code
    code = "def my_safe_metric(gt, pred):\n    import os\n    return abs(gt - pred)"
    
    # 1. Test direct Base64 encoding (simulating the checkbox/JS flow)
    encoded_code = "BASE64:" + base64.b64encode(code.encode('utf-8')).decode('utf-8')
    
    payload = {
        'name': 'test_dlp_metric',
        'description': 'Verification for DLP-safe upload',
        'python_code': encoded_code,
        'is_base64': '1'  # This flag isn't strictly needed anymore but good for testing
    }
    
    print(f"Uploading obfuscated code: {encoded_code[:30]}...")
    
    try:
        # We use a session to handle the redirect and check flash messages if possible
        # but here we just check if it returns 302 (redirect to metrics_view)
        response = requests.post(url, data=payload, allow_redirects=False)
        
        if response.status_code == 302:
            print("SUCCESS: Metric creator redirected successfully.")
        else:
            print(f"FAILED: Unexpected status code {response.status_code}")
            print(response.text[:500])
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_dlp_safe_upload()
