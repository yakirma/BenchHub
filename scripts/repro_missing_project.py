
import requests

BASE_URL = "http://127.0.0.1:6060"

def test_legacy_url():
    # Try to access a leaderboard without project prefix
    url = f"{BASE_URL}/leaderboard/1"
    print(f"Testing Legacy URL: {url}")
    try:
        response = requests.get(url, allow_redirects=False)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 404:
            print("Confirmed: Legacy URL returns 404 Not Found.")
        elif response.status_code == 200:
            print("Unexpected: Legacy URL works (200 OK).")
        elif response.status_code == 302:
            print(f"Redirected to: {response.headers.get('Location')}")
        else:
            print(f"Got unexpected status: {response.status_code}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_legacy_url()
