import numpy as np
import os

def calc_entropy(counts):
     if len(counts) == 0: return 0.0
     c = counts[counts > 0]
     if c.sum() > 0:
         p = c / c.sum()
         res = float(-np.sum(p * np.log2(p)))
         print(f"DEBUG: calc_entropy input sum={c.sum()} shape={counts.shape} non-zero={len(c)} result={res}")
         return res
     return 0.0

file_path = './uploads/submissions/4/hist_filtered/sample1.npz'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

try:
    with np.load(file_path) as data:
        print(f"Keys: {list(data.keys())}")
        counts = data['counts']
        print(f"Counts type: {type(counts)}")
        print(f"Counts shape: {counts.shape}")
        print(f"Counts sample: {counts[:10]}")
        entropy = calc_entropy(counts)
        print(f"Entropy: {entropy}")
except Exception as e:
    print(f"Error: {e}")
