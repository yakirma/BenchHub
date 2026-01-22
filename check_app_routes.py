#!/usr/bin/env python3
"""Directly import and test app.py to see what routes exist"""
import sys
sys.path.insert(0, '/Users/ymatari/Git/dtof_benchmarking')

print("Importing app...")
from app import app

print("\nAll routes:")
for rule in sorted(app.url_map.iter_rules(), key=lambda r: str(r)):
    print(f"  {str(rule.rule):50} {list(rule.methods - {'HEAD', 'OPTIONS'})}")

print("\nSearching for specific routes:")
print(f"  /test_ping: {'FOUND' if any('/test_ping' in str(r) for r in app.url_map.iter_rules()) else 'NOT FOUND'}")
print(f"  /api/dataset/upload: {'FOUND' if any('/api/dataset/upload' in str(r) for r in app.url_map.iter_rules()) else 'NOT FOUND'}")
