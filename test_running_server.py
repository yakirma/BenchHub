#!/usr/bin/env python
"""Query the running server for its routes"""
import requests

# Try different endpoints to see what works
endpoints = [
    "http://localhost:6060/",
    "http://localhost:6060/test_ping",
    "http://localhost:6060/api/dataset/upload",
]

print("Testing endpoints on running server:\n")
for url in endpoints:
    try:
        # Try GET first
        resp = requests.get(url, timeout=2)
        print(f"GET {url}")
        print(f"  Status: {resp.status_code}")
        if resp.status_code != 404:
            print(f"  Body: {resp.text[:100]}")
    except Exception as e:
        print(f"GET {url}")
        print(f"  Error: {e}")
    print()

# Also try POST for the upload endpoint
print("Testing POST to /api/dataset/upload:")
try:
    resp = requests.post("http://localhost:6060/api/dataset/upload", timeout=2)
    print(f"  Status: {resp.status_code}")
    print(f"  Body: {resp.text[:200]}")
except Exception as e:
    print(f"  Error: {e}")
