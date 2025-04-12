
#dataset.py
import os
import json
import torch
from torch.utils.data import Dataset


class OpinionQADataset(Dataset):
    def __init__(self, sample_path: str, activation_dir: str):
        """
        :param sample_path: path to preprocessed samples (JSONL)
        :param activation_dir: directory of Gemma activations (each QID as a .json file)
        """
        # 加载样本
        with open(sample_path, "r") as f:
            self.samples = [json.loads(line.strip()) for line in f if line.strip()]

        self.activation_dir = activation_dir

        # 构建 group vocab
        self.group_vocab = {}
        for sample in self.samples:
            for k, v in sample["group_features"].items():
                key = f"{k}:{v}"
                if key not in self.group_vocab:
                    self.group_vocab[key] = len(self.group_vocab)

        # 读取所有 activations（一次性加载为 Tensor）
        self.qid_to_activation = {}
        for fname in os.listdir(activation_dir):
            if fname.endswith(".json"):
                with open(os.path.join(activation_dir, fname), "r") as f:
                    item = json.load(f)
                    self.qid_to_activation[item["qid"]] = torch.tensor(item["activation"], dtype=torch.float)

        # 构建 aligned activation tensor
        self.activations = []
        for s in self.samples:
            qid = s["qid"]
            if qid not in self.qid_to_activation:
                raise ValueError(f"Missing activation for qid={qid}")
            self.activations.append(self.qid_to_activation[qid])
        self.activations = torch.stack(self.activations)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        activation = self.activations[idx]

        # 群体 one-hot
        group_vec = torch.zeros(len(self.group_vocab))
        for k, v in sample["group_features"].items():
            key = f"{k}:{v}"
            if key in self.group_vocab:
                group_vec[self.group_vocab[key]] = 1.0

        label = torch.tensor(sample["label_value"], dtype=torch.float)

        return {
            "activation": activation,
            "group_feats": group_vec,
            "activation_with_group": torch.cat([activation, group_vec], dim=-1),
            "group_id": torch.argmax(group_vec).long(),  # for embedding layer
            "label": label,
        }

