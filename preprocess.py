import os
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np
import joblib

# === 路径设置 ===
data_root = "data/human_resp"
output_dir = os.path.join("data", "preprocessed")
os.makedirs(output_dir, exist_ok=True)

steer_path = os.path.join("data/model_input", "steer-qa.csv")
steer_df = pd.read_csv(steer_path, sep="\t")

# === 群体特征元信息 ===
allowed_group_pairs = set((row["md"], str(row["subgroup"])) for _, row in steer_df.iterrows())
steer_group_keys = sorted({row["md"] for _, row in steer_df.iterrows()})

# === 遍历所有 survey 文件夹 ===
all_samples = []
for folder_name in os.listdir(data_root):
    folder_path = os.path.join(data_root, folder_name)
    if not os.path.isdir(folder_path): continue

    info_path = os.path.join(folder_path, "info.csv")
    responses_path = os.path.join(folder_path, "responses.csv")
    if not all(os.path.exists(p) for p in [info_path, responses_path]): continue

    info = pd.read_csv(info_path)
    responses = pd.read_csv(responses_path)

    for qid in tqdm(info["key"].unique().tolist(), desc=f"Processing {folder_name}"):
        if qid not in responses.columns: continue
        row = info[info["key"] == qid].iloc[0]
        question = row["question"]
        option_mapping = eval(row["option_mapping"]) if isinstance(row["option_mapping"], str) else {}
        text_to_value = {v: float(k) for k, v in option_mapping.items()}

        df = responses[[qid] + steer_group_keys].dropna(subset=[qid])
        for _, r in df.iterrows():
            answer = str(r[qid]).strip()
            if answer not in text_to_value: continue
            label_value = text_to_value[answer]
            valid_feats = {g: r[g] for g in steer_group_keys if pd.notna(r[g]) and (g, str(r[g])) in allowed_group_pairs}
            if not valid_feats: continue
            all_samples.append({
                "qid": qid,
                "question": question,
                "label_value": label_value,
                "group_features": valid_feats,
            })

# === 保存 train/test.json ===
train, test = train_test_split(all_samples, test_size=0.2, random_state=42)

with open(os.path.join(output_dir, "train.json"), "w") as f:
    for x in train: f.write(json.dumps(x) + "\n")

with open(os.path.join(output_dir, "test.json"), "w") as f:
    for x in test: f.write(json.dumps(x) + "\n")

def encode_group_features(samples, encoder=None, is_train=True):
    group_dicts = [x["group_features"] for x in samples]
    group_df = pd.DataFrame(group_dicts).fillna("Unknown")

    if is_train:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        group_encoded = encoder.fit_transform(group_df)
        joblib.dump(encoder, os.path.join(output_dir, "encoder.pkl"))
    else:
        # 将 test 的列强制对齐成和训练阶段一样的顺序
        expected_cols = encoder.feature_names_in_
        group_df = group_df.reindex(columns=expected_cols, fill_value="Unknown")
        group_encoded = encoder.transform(group_df)

    group_vectors = torch.tensor(group_encoded, dtype=torch.float32)
    _, group_labels_np = np.unique(group_encoded, axis=0, return_inverse=True)
    group_labels = torch.tensor(group_labels_np, dtype=torch.long)

    return group_vectors, group_labels, encoder

group_vectors_train, group_labels_train, encoder = encode_group_features(train, is_train=True)
group_vectors_test, group_labels_test, _ = encode_group_features(test, encoder=encoder, is_train=False)

# === 保存 .pt 文件 ===
torch.save(group_vectors_train, os.path.join(output_dir, "group_vectors.pt"))
torch.save(group_labels_train, os.path.join(output_dir, "group_labels.pt"))
torch.save(group_vectors_test, os.path.join(output_dir, "group_vectors_test.pt"))
torch.save(group_labels_test, os.path.join(output_dir, "group_labels_test.pt"))
