import requests
from urllib.parse import quote

API_BASE_URL = "http://127.0.0.1:6060"
leaderboard_name = "Target 1" # Example with space

# 1. Unquoted (likely fails if spaces are present)
info_url_unquoted = f"{API_BASE_URL}/api/leaderboard/by_name/{leaderboard_name}/info"
print(f"Testing unquoted URL: {info_url_unquoted}")
try:
    resp = requests.get(info_url_unquoted)
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text}")
except Exception as e:
    print(f"Error unquoted: {e}")

# 2. Quoted
info_url_quoted = f"{API_BASE_URL}/api/leaderboard/by_name/{quote(leaderboard_name)}/info"
print(f"\nTesting quoted URL: {info_url_quoted}")
try:
    resp = requests.get(info_url_quoted)
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text}")
except Exception as e:
    print(f"Error quoted: {e}")
