#merge_chunks.py
import torch
import os
import re

chunk_dir = "activation_chunks_test"
final_tensor = []
skipped_files = []

# 识别 chunk 文件编号（比如 chunk_0_123.pt → 123）
def get_chunk_index(filename):
    match = re.search(r"chunk_(\d+)_(\d+)\.pt", filename)
    return int(match.group(2)) if match else -1

chunk_files = sorted([
    f for f in os.listdir(chunk_dir) if f.endswith(".pt")
], key=get_chunk_index)

print(f"Found {len(chunk_files)} chunk files.")

for i, f in enumerate(chunk_files):
    path = os.path.join(chunk_dir, f)
    try:
        tensor = torch.load(path)
        final_tensor.append(tensor)
        print(f"[{i+1}/{len(chunk_files)}] Loaded {f} with shape {tensor.shape}")
    except Exception as e:
        print(f"[{i+1}/{len(chunk_files)}] Skipped {f} due to error: {e}")
        skipped_files.append(f)

# 保存有效合并后的 tensor
if final_tensor:
    try:
        final_tensor = torch.cat(final_tensor, dim=0)
        torch.save(final_tensor, "gemma_activations_test.pt")
        print(f"\nSaved gemma_activations.pt with shape: {final_tensor.shape}")
    except Exception as e:
        print(f"\nFailed to save final tensor: {e}")
else:
    print("\nNo valid chunks loaded. Nothing was saved.")

# 自动生成补跑命令
if skipped_files:
    print("\nSkipped files:")
    for f in skipped_files:
        print(f"  - {f}")
    
    print("\nSuggested rerun commands:")
    for f in skipped_files:
        parts = f.split("_")
        start = int(parts[1])
        idx = int(parts[2].split(".")[0])
        chunk_size = 64  # 默认每个 chunk 保存 8 个样本
        begin = idx * chunk_size
        end = begin + chunk_size
        print(f"TorchDistributed script command:")
        print(f"python3 -m torch.distributed.run --nproc_per_node=3 extract_activations.py --start {begin} --end {end}")
