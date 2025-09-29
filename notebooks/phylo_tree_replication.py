# %%
from datasets import load_dataset
from pprint import pprint

# Example: Load a JSON dataset from the Hugging Face Hub
# Replace 'username/dataset_name' with the actual dataset path
dataset = load_dataset(
    'json',
    data_files='https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/pretraining_or_both_phases/gtdb_v220_imgpr/data_gtdb_train_chunk1.jsonl.gz'
)

# If the dataset is hosted on the Hugging Face Hub, you can use:
# dataset = load_dataset('username/dataset_name')

# Access the data
pprint(dataset['train'][0]) # record, text


# %% - Get items with longest sequences
# TODO: consider how sampling might be biased by taking top n sorted sequences

print(len(dataset["train"]), "\n")
# Sort the dataset by text length (descending) and sample from the longest
num_items = 10
# Get all items and their indices
items_with_length = [
    (i, item.get('text', '')) for i, item in enumerate(dataset['train'])
]
# Sort by text length descending
items_with_length.sort(key=lambda x: len(x[1]), reverse=True)
# Take the top num_items
for idx, text in items_with_length[:num_items]:
    pipe_count = text.count('|')
    print(f"Item {idx}: {pipe_count} pipe characters. Text length: {len(text)}")

# %%  - sampling
# Configurable variables
NUM_SPECIES = 10  # Number of species to sample (for demo; set higher for real use)
REGION_LENGTH = 4000  # Length of each genomic region to sample (bp)
AVERAGE_OVER_LAST_BP = 2000  # Only average activations over the last N bp of each region

import random

def tiling_helper(
    text: str,
    sample_region_length: int,
    coverage_fraction: float = 0.05,
    max_attempts: int = 10000
) -> list[str]:
    """
    Randomly sample non-overlapping regions from the input text until at least
    `coverage_fraction` (default 5%) of the text is covered.
    Returns a list of non-overlapping sampled regions.
    """
    text_length = len(text)
    if text_length < sample_region_length:
        print(f"Warning: Text is shorter than sample_region_length ({text_length} < {sample_region_length})")
        return []

    target_coverage = int(text_length * coverage_fraction)
    covered_bp = 0
    used_intervals = []
    regions = []
    attempts = 0

    # Helper to check overlap
    def overlaps(start, end, intervals):
        for (used_start, used_end) in intervals:
            if not (end <= used_start or start >= used_end):
                return True
        return False

    while covered_bp < target_coverage and attempts < max_attempts:
        start_pos = random.randint(0, text_length - sample_region_length)
        end_pos = start_pos + sample_region_length

        if not overlaps(start_pos, end_pos, used_intervals):
            region = text[start_pos:end_pos]
            regions.append(region)
            used_intervals.append((start_pos, end_pos))
            covered_bp += sample_region_length
        attempts += 1

        # If we've exhausted all possible non-overlapping regions, break
        if len(used_intervals) >= (text_length // sample_region_length):
            break

    if covered_bp < target_coverage:
        print(f"Warning: Only able to cover {covered_bp} bp out of requested {target_coverage} bp ({coverage_fraction*100:.1f}% of text).")

    return regions

# regions_list = []
# for idx, text in items_with_length[:num_items]:
#     regions_list.append(tiling_helper(text, sample_region_length=REGION_LENGTH))
#     print(f"Item {idx}: {len(regions_list[-1])} regions sampled.")

# %%
import torch
from evo2 import Evo2

evo2_model = Evo2('evo2_7b')
sequence = 'ACGT'

input_ids = torch.tensor(
    evo2_model.tokenizer.tokenize(sequence),
    dtype=torch.int,
).unsqueeze(0).to('cuda:0')

layer_name = 'blocks.28.mlp.l3'

outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])

print('Embeddings shape: ', embeddings[layer_name].shape)
# %%

# Save the loaded dataset to disk for local caching
dataset.save_to_disk("data/gtdb_v220_imgpr_hf_cache")
print("Dataset cached locally at data/gtdb_v220_imgpr_hf_cache")

# %%
pprint(dataset["train"].keys())
# %%
print(dataset)
# %%
print(dataset["train"])
# %%
midtrain_data_sample = load_dataset(
    'json',
    data_files='https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/midtraining_specific/imgpr/data_imgpr_test_chunk1.jsonl.gz'
)
pprint(dataset['train'][0])
# %%

regions_list = []
for idx, text in items_with_length[:num_items]:
    regions_list.append(tiling_helper(text, sample_region_length=REGION_LENGTH))
    print(f"Item {idx}: {len(regions_list[-1])} regions sampled.")
print(len(regions_list))
# %%
print(regions_list[0])
print(len(regions_list[0]))
# %%
print(evo2_model.model)
# %%
# inputs_by_species = []
mean_embeddings = []

for genome in regions_list: # genome is a list of regions
    # for region in regions_list:
    input_ids = torch.tensor(
        evo2_model.tokenizer.tokenize(genome),
        dtype=torch.int,
    ).unsqueeze(0).to('cuda:0')
    tokenized_input_ids = input_ids

    layer_name = 'blocks.24.mlp.l3'

    outputs, embeddings = evo2_model(tokenized_input_ids, return_embeddings=True, layer_names=[layer_name])

    print('Embeddings shape: ', embeddings[layer_name].shape)
    mean_embeddings.append(embeddings[layer_name][AVERAGE_OVER_LAST_BP:].mean(dim=0).mean())

print(mean_embeddings)
# %%
print(len(mean_embeddings))
# %%
print(mean_embeddings[0])
# %%