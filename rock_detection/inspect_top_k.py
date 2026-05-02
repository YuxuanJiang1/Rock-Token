import torch
from transformers import AutoTokenizer

# --- Configuration ---
PT_FILE = "/workspace/aggregated_rock_token_stats.pt"
TOKENIZER_ID = "RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k"
TOP_K = 50

# --- Load ---
print(f"Loading data from {PT_FILE}...")
data = torch.load(PT_FILE, map_location="cpu")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

average_kl = data["average_kl"]
token_frequencies = data["frequencies"]

# --- Top-K by Average KL ---
valid_indices = (token_frequencies > 0).nonzero(as_tuple=True)[0]
valid_avg_kl = average_kl[valid_indices]

top_k_relative = torch.topk(valid_avg_kl, k=min(TOP_K, len(valid_indices)))
top_k_token_ids = valid_indices[top_k_relative.indices]
top_k_avg_kl = top_k_relative.values
top_k_freqs = token_frequencies[top_k_token_ids]

print(f"\nTop {TOP_K} tokens by average KL divergence:")
print(f"{'Rank':<6} {'Token ID':<12} {'Token String':<30} {'Avg KL':<12} {'Frequency'}")
print("-" * 75)
for rank, (tid, kl, freq) in enumerate(zip(top_k_token_ids.tolist(), top_k_avg_kl.tolist(), top_k_freqs.tolist()), 1):
    token_str = repr(tokenizer.decode([tid]))
    print(f"{rank:<6} {tid:<12} {token_str:<30} {kl:<12.6f} {freq}")
