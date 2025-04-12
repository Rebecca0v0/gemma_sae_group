import torch
import os

chunk_dir = "activation_chunks"
chunk_files = sorted([
    f for f in os.listdir(chunk_dir) if f.endswith(".pt")
], key=lambda x: int(x.split("_")[-1].split(".")[0]))

final_tensor = []
for f in chunk_files:
    final_tensor.append(torch.load(os.path.join(chunk_dir, f)))

final_tensor = torch.cat(final_tensor, dim=0)
torch.save(final_tensor, "gemma_activations.pt")
print("Saved gemma_activations.pt with shape:", final_tensor.shape)
