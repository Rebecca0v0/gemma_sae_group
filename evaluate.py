import os, json, torch
import torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import wandb
from torch.utils.data import TensorDataset, DataLoader
from train import FastAutoencoder, Config, make_torch_comms

# === 配置加载 ===
with open("checkpoints/config.json") as f:
    cfg_dict = json.load(f)
cfg_dict["n_op_shards"] = 1
cfg_dict["n_replicas"] = 1
cfg = Config(**cfg_dict)

device = "cuda" if torch.cuda.is_available() else "cpu"
activation_path = "gemma_activations_test.pt"
group_vector_path = "data/preprocessed/group_vectors_test.pt"
group_label_path = "data/preprocessed/group_labels_test.pt"

# === 启动 WandB
wandb.init(project="OpinionQA-AE", name="eval", job_type="evaluation")

# === 加载数据 ===
x_tensor = torch.load(activation_path).to(device)            # [N, 2048]
group_vectors = torch.load(group_vector_path).to(device)     # [N, 30]
group_labels = torch.load(group_label_path)                  # [N]

# === 加载模型权重并推断维度 ===
state_dict = torch.load("sparse_ae.pt")
activation_dim = x_tensor.shape[1]                            # 2048
group_dim = group_vectors.shape[1]                            # 30
input_dim = activation_dim + group_dim                        # 2078

print("✔ encoder.weight:", state_dict["encoder.weight"].shape)
print("✔ decoder.weight:", state_dict["decoder.weight"].shape)
print("✔ pre_bias shape:", state_dict["pre_bias"].shape)

# === 构造模型 ===
ae = FastAutoencoder(
    n_dirs_local=state_dict["encoder.weight"].shape[0],       # 10922
    d_model=input_dim,                                        # 2078
    group_dim=group_dim,                                      # 30
    k=cfg.k,
    auxk=cfg.auxk,
    dead_steps_threshold=cfg.dead_toks_threshold // cfg.bs,
    comms=make_torch_comms(1, 1),
).to(device)

ae.load_state_dict(state_dict, strict=True)
ae.eval()

# === 拼接 x 与 group_vector，构建 DataLoader ===
x_input = torch.cat([x_tensor, group_vectors], dim=1)  # [N, 2078]
dataset = TensorDataset(x_input, x_tensor, group_labels)
dataloader = DataLoader(dataset, batch_size=8192, shuffle=False)

# === 批处理评估 ===
all_latents, all_recons, all_labels = [], [], []
mse_total, n_total = 0.0, 0

with torch.no_grad():
    for x_input_batch, x_orig_batch, label_batch in dataloader:
        x_input_batch = x_input_batch.to(device)
        x_orig_batch = x_orig_batch.to(device)

        x_group = x_input_batch[:, activation_dim:]
        recons, info = ae(x_input_batch, group_vec=x_group)

        recons_x = recons[:, :activation_dim]
        mse = F.mse_loss(recons_x, x_orig_batch, reduction="sum").item()
        mse_total += mse
        n_total += x_orig_batch.size(0)

        all_latents.append(info["latents"].cpu())
        all_recons.append(recons_x.cpu())
        all_labels.append(label_batch.cpu())

mse_recon = mse_total / n_total
wandb.log({"mse_reconstruction": mse_recon})

# === PCA 可解释性分析 ===
latents = torch.cat(all_latents).numpy()
pca = PCA(n_components=10).fit(latents)
explained_var = pca.explained_variance_ratio_
wandb.log({"pca_variance_ratio_sum_top10": explained_var.sum()})
plt.plot(np.cumsum(explained_var))
plt.title("Cumulative Explained Variance (Latents)")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance Ratio")
plt.grid(True)
wandb.log({"pca_curve": wandb.Image(plt)})
plt.clf()

# === 下游分类精度 ===
labels = torch.cat(all_labels).numpy()
clf = LogisticRegression(max_iter=500).fit(latents, labels)
preds = clf.predict(latents)
acc = accuracy_score(labels, preds)
wandb.log({"downstream_accuracy": acc})

# === Group Steering Δ（重构偏差）===
x_tensor_cpu = torch.cat([d[1] for d in dataset])
recons_tensor = torch.cat(all_recons)
group_ids = labels
delta_list = []

for gid in np.unique(group_ids):
    idx = group_ids == gid
    group_mean = x_tensor_cpu[idx].mean(dim=0)
    recon_mean = recons_tensor[idx].mean(dim=0)
    delta = F.mse_loss(recon_mean, group_mean, reduction="sum").item()
    delta_list.append(delta)

delta_score = np.mean(delta_list)
wandb.log({"group_delta": delta_score})
plt.bar(np.arange(len(delta_list)), delta_list)
plt.title("Per-Group Δ (MSE)")
plt.xlabel("Group ID")
plt.ylabel("MSE between mean(x) and mean(recon)")
wandb.log({"group_delta_bar": wandb.Image(plt)})

print("✅ Evaluation completed.")
