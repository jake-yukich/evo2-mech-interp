# %%
%load_ext autoreload
%autoreload 2
# %%
import gc
import os
import sys
import random
from typing import Literal
sys.path.append("..")  # For imports from parent dir
from utils.distances import build_knn_graph, geodesic_distance_matrix, cosine_similarity_matrix, mp_phylogenetic_distance_matrix
from utils.data import load_data_from_hf, remove_tags, preprocess_gtdb_sequences, add_gtdb_accession
from utils.phylogenetics import get_tag_to_gtdb_accession_map
from utils.sampling import sample_genome
from utils.inference import get_mean_embeddings, batch_tokenize
from utils.visualization import umap_reduce_3d, plot_umap_3d, plot_distance_scatter

import numpy as np
import torch
from evo2 import Evo2
from tqdm import tqdm
import pandas as pd

random.seed(42)
# Enable optimizations
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
if hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = (
        True  # Use TF32 for faster matmul on A100/H100
    )

assert os.getcwd().endswith("evo2-mech-interp/notebooks"), "Run from notebooks/ directory"


NUM_SPECIES = 64  # Number of species to sample (for demo; set higher for real use)
NUM_SAMPLES = 10  # Number of samples per genome
COVERAGE_FRACTION = 0.05  # Fraction of genome to cover with samples
REGION_LENGTH = 4000  # Length of each genomic region to sample (bp)
AVERAGE_OVER_LAST_BP = (
    2000  # Only average activations over the last N bp of each region
)
D_MODEL_7B = 4096
D_MODEL_1B = 1920
MODEL: Literal["1b", "7b"] = "7b" 
D_MODEL = D_MODEL_1B if MODEL == "1b" else D_MODEL_7B if MODEL == "7b" else None
BATCH_SIZE = 48 if MODEL == "1b" else 8  # Start here and increase if possible
RANDOM_SEED = 42
LAYER_NAME = "blocks.24.mlp.l3"
REMOVE_TAGS = True

SAMPLING_CONFIG = {
    "num_samples": 5,
    "coverage_fraction": None,
}
SAMPLING_STR = f"num_samples_{NUM_SAMPLES}" if SAMPLING_CONFIG["num_samples"] is not None else f"coverage_frac_{COVERAGE_FRACTION}"
CACHE_PATH = f"data/embeddings/model_{MODEL}/{SAMPLING_STR}/layer_{LAYER_NAME}/low_species_coverage_5p/"

os.makedirs(CACHE_PATH, exist_ok=True)


# %%
# data_files = [
#     "https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/pretraining_or_both_phases/gtdb_v220_imgpr/data_gtdb_train_chunk1.jsonl.gz"
# ]
# %% - LOAD DATA
data_files = [
    "https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/midtraining_specific/gtdb_v220_stitched/data_gtdb_train_chunk1.jsonl.gz"
]

df = load_data_from_hf(data_files)

# %% - Get items with longest sequences
# TODO: consider how sampling might be biased by taking top n sorted sequences

df = preprocess_gtdb_sequences(df, NUM_SAMPLES * REGION_LENGTH, subset=NUM_SPECIES, random_seed=RANDOM_SEED)
df = add_gtdb_accession(df, get_tag_to_gtdb_accession_map())
df = df.dropna(subset=["gtdb_accession"]).reset_index(drop=True)
df.to_csv(f"{CACHE_PATH}/genomes_metadata.csv", index=False)
df.head()
# %%
# Sample and save to new df where each row is a sampled region, referencing original genome
samples = {
    "genome_idx": [],
    "sample": [],
}
for row in df.itertuples():
    sampled_regions = sample_genome(
        row.sequence, 
        sample_region_length=REGION_LENGTH, 
        **SAMPLING_CONFIG
    )
    samples["genome_idx"].extend([row.Index] * len(sampled_regions))
    samples["sample"].extend(sampled_regions)
    if len(set(samples["genome_idx"])) >= NUM_SPECIES:
        break

samples_df = pd.DataFrame(samples)
samples_df.to_csv(f"{CACHE_PATH}/sampled_regions.csv", index=False)
samples_df.head()

# %% - Load model
evo2_model = Evo2("evo2_1b_base") if MODEL == "1b" else Evo2("evo2_7b")
print(f"Loaded Evo2 model with {sum(p.numel() for p in evo2_model.model.parameters() if p.requires_grad):,} parameters")

# %%
torch.cuda.empty_cache()
gc.collect()

print("Pre-tokenizing all sequences...")
# Pre-tokenize all sequences to avoid repeated tokenization
tokenized_samples = []
for genome_idx, genome_df in tqdm(samples_df.groupby("genome_idx"), desc="Tokenizing"):
    tokenized_samples.append(
        batch_tokenize(
            genome_df["sample"].tolist(), 
            evo2_model,
            BATCH_SIZE
        )
    )

print("Calculating mean embeddings for all genomes...")
mean_embeddings = get_mean_embeddings(
    df,
    tokenized_samples,
    evo2_model,
    BATCH_SIZE,
    CACHE_PATH,
    D_MODEL,
    REGION_LENGTH,
    LAYER_NAME,
    AVERAGE_OVER_LAST_BP,
)


# %%
embedding_3d = umap_reduce_3d(mean_embeddings, random_state=42)

# %%
# %%
for category in ["class", "order", "family"]:
    # Render ALL points; map rare categories to "Other" for readability
    # Ensure labels length matches embedding_3d rows
    labels_series = (
        df[category]
        .fillna("Unknown")
        .replace("NONE", "Unknown")
    )
    value_counts = labels_series.value_counts()
    min_count = max(2, int(0.05 * len(labels_series)))  # at least 2, or 5%
    frequent = set(value_counts[value_counts >= min_count].index)
    if len(frequent) == 0 and len(value_counts) > 0:
        frequent = set(value_counts.head(min(10, len(value_counts))).index)
    # Truncate or pad labels_for_plot to match embedding_3d rows
    n_points = embedding_3d.shape[0]
    labels_list = [lbl if lbl in frequent else "Other" for lbl in labels_series.tolist()]
    if len(labels_list) > n_points:
        labels_for_plot = labels_list[:n_points]
    elif len(labels_list) < n_points:
        labels_for_plot = labels_list + ["Other"] * (n_points - len(labels_list))
    else:
        labels_for_plot = labels_list

    fig = plot_umap_3d(
        embedding_3d,
        title=f"3D UMAP Colored by {category.capitalize()}",
        labels=labels_for_plot,
    )
    fig.show()
    # fig.write_html(f"{CACHE_PATH}/3d_embedding_{category}.html")

# %%
# Compute phylogenetic distance matrix
phylo_distance_matrix = mp_phylogenetic_distance_matrix(df["gtdb_accession"].tolist(), "/root/evo2-mech-interp/data/gtdb/bac120_r220.tree")
torch.save(phylo_distance_matrix, f"{CACHE_PATH}/phylogenetic_distance_matrix.pt")

# %%

# Build the KNN adjacency graph and compute geodesics
adjacency_matrix = build_knn_graph(mean_embeddings, k=27, distance='cosine', weighted=True)
geo_distance_matrix = geodesic_distance_matrix(adjacency_matrix)
cos_similarity_matrix = cosine_similarity_matrix(mean_embeddings)

# %%
# Use lower triangular indices, excluding diagonal
n = cos_similarity_matrix.shape[0]
tril_indices = np.tril_indices(n, k=-1)

fig = plot_distance_scatter(
    x=phylo_distance_matrix[tril_indices].to(torch.float32).cpu().numpy(),
    y=cos_similarity_matrix[tril_indices].to(torch.float32).cpu().numpy(),
    x_label="Phylogenetic Distance",
    y_label="Cosine Similarity",
    title="Cosine Similarity vs Phylogenetic Distance"
)
fig.show()

fig = plot_distance_scatter(
    x=phylo_distance_matrix[tril_indices].to(torch.float32).cpu().numpy(),
    y=geo_distance_matrix[tril_indices],
    x_label="Phylogenetic Distance",
    y_label="Geodesic Distance",
    title="Geodesic Distance vs Phylogenetic Distance"
)
fig.show()
# %%
