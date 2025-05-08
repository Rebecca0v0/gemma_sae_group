import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse

torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

# === ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="Start index of data to process")
parser.add_argument("--end", type=int, default=-1, help="End index of data to process (exclusive)")
args = parser.parse_args()

# === CONFIG ===
model_id = "gemma-2b-local"
data_path = "data/preprocessed/train.json"
output_dir = "activation_chunks"
batch_size = 8  # Reduce if OOM
final_output = "gemma_activations.pt"

os.makedirs(output_dir, exist_ok=True)

# === GPU device per process ===
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map={"": local_rank},  # force local_rank mapping
    local_files_only=True
)
model.eval()

# === LOAD DATA ===
with open(data_path, "r") as f:
    all_samples = [json.loads(line) for line in f]

if args.end == -1:
    args.end = len(all_samples)

samples = all_samples[args.start:args.end]
questions = [sample["question"] for sample in samples]
print(f"Processing {len(questions)} questions: [{args.start}, {args.end})")

# === Hook for final layer ===
target_layer = len(model.model.layers) - 1
hidden_states_list = []

def get_hidden_hook(module, input, output):
    hidden_states_list.append(output)

hook = model.model.layers[target_layer].register_forward_hook(get_hidden_hook)

# === Resume support ===
existing_chunks = sorted([
    f for f in os.listdir(output_dir)
    if f.startswith(f"chunk_{args.start}_") and f.endswith(".pt")
])
start_idx = len(existing_chunks)
print(f"Resuming from chunk index: {start_idx}")

# === Forward Loop ===
with torch.no_grad():
    for i in tqdm(range(start_idx * batch_size, len(questions), batch_size)):
        batch = questions[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(device)

        hidden_states_list.clear()
        _ = model(**inputs)

        output = hidden_states_list[0]
        if isinstance(output, tuple):
            output = output[0]

        acts = []
        for j in range(len(batch)):
            seq_len = (inputs["attention_mask"][j] == 1).sum().item()
            last_token_state = output[j, seq_len - 1, :].cpu()
            acts.append(last_token_state)

        chunk_tensor = torch.stack(acts)
        chunk_path = os.path.join(output_dir, f"chunk_{args.start}_{i // batch_size}.pt")
        torch.save(chunk_tensor, chunk_path)
        print(f"Saved {chunk_path}")

hook.remove()
