# preprocess_test.py

import os
import json
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.preprocessing import OneHotEncoder

# === 目录配置 ===
test_path = "data/preprocessed/test.json"
output_dir = os.path.join("data", "preprocessed")

with open(os.path.join(test_path)) as f:
    test_samples = [json.loads(line) for line in f]

group_dicts = [x["group_features"] for x in test_samples]
group_df = pd.DataFrame(group_dicts).fillna("Unknown")

# 保存 one-hot 编码器用于测试时对齐
encoder = OneHotEncoder(sparse_output=False)
group_encoded = encoder.fit_transform(group_df)

# 保存向量和 label（按行哈希，或者用 category 编号）
group_vectors = torch.tensor(group_encoded, dtype=torch.float32)

# 简单分组：按 unique row 编号为群体 ID
_, group_labels_np = np.unique(group_encoded, axis=0, return_inverse=True)
group_labels = torch.tensor(group_labels_np, dtype=torch.long)

torch.save(group_vectors, os.path.join(output_dir, "group_vectors_test.pt"))
torch.save(group_labels, os.path.join(output_dir, "group_labels_test.pt"))