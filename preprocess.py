import os
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np

# ========== 基本设置 ==========
data_root = "data/human_resp"
output_dir = os.path.join("data", "preprocessed")
os.makedirs(output_dir, exist_ok=True)

steer_path = os.path.join("data/model_input", "steer-qa.csv")
steer_df = pd.read_csv(steer_path, sep="\t")

# 创建允许的群体特征值对：如 ('EDUCATION', 'College graduate/some postgrad')
allowed_group_pairs = set(
    (row["md"], str(row["subgroup"])) for _, row in steer_df.iterrows()
)

# 我们只关心 steer_qa.csv 中提到的 group key
steer_group_keys = sorted({row["md"] for _, row in steer_df.iterrows()})

# ========== 数据预处理主流程 ==========
all_samples = []

for folder_name in os.listdir(data_root):
    folder_path = os.path.join(data_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    info_path = os.path.join(folder_path, "info.csv")
    meta_path = os.path.join(folder_path, "metadata.csv")
    responses_path = os.path.join(folder_path, "responses.csv")

    if not all(os.path.exists(p) for p in [info_path, meta_path, responses_path]):
        continue

    info = pd.read_csv(info_path)
    responses = pd.read_csv(responses_path)

    question_keys = info["key"].unique().tolist()

    for qid in tqdm(question_keys, desc=f"Processing {folder_name}"):
        if qid not in responses.columns:
            continue

        row = info[info["key"] == qid].iloc[0]
        question = row["question"]
        option_mapping = eval(row["option_mapping"]) if isinstance(row["option_mapping"], str) else {}
        text_to_value = {v: float(k) for k, v in option_mapping.items()}

        selected_cols = [qid] + steer_group_keys
        df = responses[selected_cols].dropna(subset=[qid])

        for _, r in df.iterrows():
            answer = str(r[qid]).strip()
            if answer not in text_to_value:
                continue

            label_value = text_to_value[answer]

            # 只保留用户属于 steer_qa.csv 中某个群体值的特征对
            valid_feats = {
                g: r[g] for g in steer_group_keys
                if pd.notna(r[g]) and (g, str(r[g])) in allowed_group_pairs
            }
            if not valid_feats:
                continue

            sample = {
                "qid": qid,
                "question": question,
                "label_value": label_value,
                "group_features": valid_feats,
            }
            all_samples.append(sample)

# ========== 划分并保存 ==========
if len(all_samples) == 0:
    raise ValueError("[!] all_samples is empty, please check your filtering conditions or input files.")

train, test = train_test_split(all_samples, test_size=0.2, random_state=42)

with open(os.path.join(output_dir, "train.json"), "w") as f:
    for x in train:
        f.write(json.dumps(x) + "\n")

with open(os.path.join(output_dir, "test.json"), "w") as f:
    for x in test:
        f.write(json.dumps(x) + "\n")

# 加载 train.json 中所有 group features
with open(os.path.join(output_dir, "train.json")) as f:
    train_samples = [json.loads(line) for line in f]

group_dicts = [x["group_features"] for x in train_samples]
group_df = pd.DataFrame(group_dicts).fillna("Unknown")

# 保存 one-hot 编码器用于测试时对齐
encoder = OneHotEncoder(sparse_output=False)
group_encoded = encoder.fit_transform(group_df)

# 保存向量和 label（按行哈希，或者用 category 编号）
group_vectors = torch.tensor(group_encoded, dtype=torch.float32)

# 简单分组：按 unique row 编号为群体 ID
_, group_labels_np = np.unique(group_encoded, axis=0, return_inverse=True)
group_labels = torch.tensor(group_labels_np, dtype=torch.long)

torch.save(group_vectors, os.path.join(output_dir, "group_vectors.pt"))
torch.save(group_labels, os.path.join(output_dir, "group_labels.pt"))
