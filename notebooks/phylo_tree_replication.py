# %%
from datasets import load_dataset
from pprint import pprint

import random
random.seed(42)

from tqdm import tqdm

NUM_SPECIES = 1024  # Number of species to sample (for demo; set higher for real use)
NUM_SAMPLES = 10  # Number of samples per genome
REGION_LENGTH = 4000  # Length of each genomic region to sample (bp)
AVERAGE_OVER_LAST_BP = 2000  # Only average activations over the last N bp of each region

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

# Get all items and their indices
items_with_length = [
    (i, item.get('text', '')) for i, item in enumerate(dataset['train'])
]
# Sort by text length descending
# Filter for sequence lengths greater than num_samples * region_length
min_length = NUM_SAMPLES * REGION_LENGTH
items_with_length = [item for item in items_with_length if len(item[1]) > min_length]
# items_with_length.sort(key=lambda x: len(x[1]), reverse=True)
random.shuffle(items_with_length)
# Take the top num_items
for idx, text in items_with_length[:NUM_SPECIES]:
    pipe_count = text.count('|')
    print(f"Item {idx}: {pipe_count} pipe characters. Text length: {len(text)}")

# %%  - SAMPLING

def tiling_helper(
    text: str,
    sample_region_length: int,
    coverage_fraction: float | None = None,
    num_samples: int | None = None,
    max_attempts: int = 10000
) -> list[str]:
    """
    Randomly sample non-overlapping regions from the input text until at least
    `coverage_fraction` (default 5%) of the text is covered.
    Returns a list of non-overlapping sampled regions.
    """
    print(coverage_fraction, num_samples)
    if (coverage_fraction is not None) and (num_samples is not None):
        raise ValueError("Specify either coverage_fraction or num_samples, not both.")
    if (coverage_fraction is None) and (num_samples is None):
        raise ValueError("Specify either coverage_fraction or num_samples.")
    text_length = len(text)
    if text_length < sample_region_length:
        print(f"Warning: Text is shorter than sample_region_length ({text_length} < {sample_region_length})")
        return []

    target_coverage = int(text_length * coverage_fraction) if coverage_fraction is not None else num_samples * sample_region_length
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

    if coverage_fraction is not None and covered_bp < target_coverage:
        print(f"Warning: Only able to cover {covered_bp} bp out of requested {target_coverage} bp ({coverage_fraction*100:.1f}% of text).")

    return regions

# regions_list = []
# for idx, text in items_with_length[:num_items]:
#     regions_list.append(tiling_helper(text, sample_region_length=REGION_LENGTH))
#     print(f"Item {idx}: {len(regions_list[-1])} regions sampled.")

# %%
import torch
from evo2 import Evo2
MODEL = "1b"  # "1b" or "7b"

evo2_model = Evo2('evo2_1b_base') if MODEL == "1b" else Evo2('evo2_7b_base')
# sequence = 'ACGT'

# input_ids = torch.tensor(
#     evo2_model.tokenizer.tokenize(sequence),
#     dtype=torch.int,
# ).unsqueeze(0).to('cuda:0')

# layer_name = 'blocks.17.mlp.l3'

# outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])

# print('Embeddings shape: ', embeddings[layer_name].shape)
# %%

# Save the loaded dataset to disk for local caching
# dataset.save_to_disk("data/gtdb_v220_imgpr_hf_cache")
# print("Dataset cached locally at data/gtdb_v220_imgpr_hf_cache")

# %%
print(dataset)
# %%
print(dataset["train"])
# %%
# midtrain_data_sample = load_dataset(
#     'json',
#     data_files='https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/midtraining_specific/imgpr/data_imgpr_test_chunk1.jsonl.gz'
# )
# pprint(dataset['train'][0])
# %%

regions_list = []
for idx, text in items_with_length[:NUM_SPECIES]:
    regions_list.append(tiling_helper(
            text, 
            sample_region_length=REGION_LENGTH,
            num_samples=NUM_SAMPLES  # Fixed number of samples per genome for testing
        )
    )
    print(f"Item {idx}: {len(regions_list[-1])} regions sampled.")
print(len(regions_list))
# %%
print(regions_list[0])
print(len(regions_list[0]))
# %%
print(evo2_model.model)
# %% - GET SEQUENCE EMBEDDINGS
import gc
from itertools import batched
torch.cuda.empty_cache()
gc.collect()
from time import perf_counter
from tqdm import tqdm

# Enable optimizations
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
if hasattr(torch.backends.cuda, 'matmul'):
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul on A100/H100

D_MODEL_7B = 4096
D_MODEL_1B = 1920
D_MODEL = D_MODEL_1B if MODEL == "1b" else D_MODEL_7B
# Optimized batch sizes - increase gradually until you hit GPU memory limits
BATCH_SIZE = 48 if MODEL == "1b" else 8  # Start here and increase if possible
# inputs_by_species = []
mean_embeddings = []
layer_name = 'blocks.24.mlp.l3'  # Move outside loop

print("Pre-tokenizing all sequences...")
# Pre-tokenize all sequences to avoid repeated tokenization
tokenized_regions_list = []
for genome in tqdm(regions_list, desc="Tokenizing"):
    tokenized_genome = [evo2_model.tokenizer.tokenize(sample) for sample in genome]
    tokenized_regions_list.append(tokenized_genome)

print("Processing embeddings...")
with torch.no_grad():
    for genome_idx, tokenized_genome in enumerate(tqdm(tokenized_regions_list, desc="Processing genomes")):
        genome_start = perf_counter()
        sample_embeddings = []
        
        for batch in batched(tokenized_genome, BATCH_SIZE):
            # Convert pre-tokenized batch to tensor
            input_ids = torch.tensor(batch, dtype=torch.int).to('cuda:0')
            input_ids = input_ids.unsqueeze(0) if input_ids.ndim == 1 else input_ids
            assert input_ids.shape == (len(batch), REGION_LENGTH), f"{input_ids.shape = }"
        
            outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
            
            sample_embeddings.append(embeddings[layer_name].cpu())  # Move to CPU immediately to save GPU memory
        
            # Clear GPU cache periodically
            if len(sample_embeddings) % 10 == 0:
                torch.cuda.empty_cache()

        # print(f"Genome {genome_idx}: time taken for {len(tokenized_genome)} samples (s): {(perf_counter() - genome_start):.3f}")
        
        # Process on CPU to save GPU memory
        sample_embeddings = torch.cat(sample_embeddings, dim=0)
        assert sample_embeddings.shape == (len(tokenized_genome), REGION_LENGTH, D_MODEL), f"Expected: {(len(tokenized_genome), REGION_LENGTH, D_MODEL)}, Got: {sample_embeddings.shape}"
        # Fixed: Use negative indexing to get the last AVERAGE_OVER_LAST_BP tokens
        genome_embedding = sample_embeddings[:, -AVERAGE_OVER_LAST_BP:, :].mean(dim=(0, 1))
        torch.save(genome_embedding, f"data/embeddings/genome_embedding_{genome_idx}.pt")
        assert genome_embedding.shape == (D_MODEL,)
        mean_embeddings.append(genome_embedding)

print(mean_embeddings)
# %%
# %%
# load embeddings from disk
import os
mean_embeddings = []
for filename in sorted(os.listdir("data/embeddings")):
    if filename.endswith(".pt"):
        embedding = torch.load(os.path.join("data/embeddings", filename))
        mean_embeddings.append(embedding)
# %%
print(len(mean_embeddings[0]))
# %%
print(len(mean_embeddings))  # list of length num_items, each item is a tensor of shape (D_MODEL,)
# %%
mean_embeddings_tensor = torch.stack(mean_embeddings, dim=0)
# fit the umap
print(mean_embeddings_tensor.shape)  # (num_items, D_MODEL)
# %%
import umap
# from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
reducer = umap.UMAP(n_components=3, random_state=42)
embedding_3d = reducer.fit_transform(mean_embeddings_tensor.to(torch.float32).cpu().numpy())
print(embedding_3d.shape)  # (num_items, 3)

# plot the 3D embedding using plotly
import ipympl
import plotly.express as px
import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter3d(
    x=embedding_3d[:, 0],
    y=embedding_3d[:, 1],
    z=embedding_3d[:, 2],
    mode='markers',
    marker=dict(
        size=8,
        color='blue',
        opacity=0.7
    ),
    text=[f'Genome {i}' for i in range(len(embedding_3d))],
    hovertemplate='<b>%{text}</b><br>UMAP 1: %{x}<br>UMAP 2: %{y}<br>UMAP 3: %{z}<extra></extra>'
)])

fig.update_layout(
    title='3D UMAP of Genome Embeddings',
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3'
    ),
    width=800,
    height=600
)

fig.show()
# %%
records = [item.get('record', '') for i, item in enumerate(dataset['train']) if i < 20]
# %%
print(records)
# %%
%autoreload 2
from id2taxonomy import get_taxonomy_from_accession
df = get_taxonomy_from_accession(records[:10])
df
# %%
# %%
df.empty

# %%
