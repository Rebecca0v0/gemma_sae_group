import os
import re
import torch

chunk_dir = "activation_chunks"
batch_size = 8
sample_indices = set()
duplicates = []

for f in os.listdir(chunk_dir):
    match = re.match(r"chunk_(\d+)_(\d+)\.pt", f)
    if not match:
        continue
    start, idx = map(int, match.groups())
    s_idx = start + idx * batch_size
    e_idx = s_idx + batch_size
    for i in range(s_idx, e_idx):
        if i in sample_indices:
            duplicates.append((f, i))
        sample_indices.add(i)

if duplicates:
    print("❌ Found duplicate sample indices:")
    for f, i in duplicates:
        print(f"  - Index {i} in file {f}")
else:
    print("✅ No duplicate sample indices found.")
