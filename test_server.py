#!/usr/bin/env python
"""Test if the running server has the route"""
import requests

url = "http://localhost:6060/api/dataset/upload"

# Test with a simple GET to see if route exists
print(f"Testing {url}...")
try:
    response = requests.get(url)
    print(f"GET Status: {response.status_code}")
    print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"Error: {e}")

# Also test /test_ping
print("\nTesting /test_ping...")
try:
    response = requests.get("http://localhost:6060/test_ping")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
