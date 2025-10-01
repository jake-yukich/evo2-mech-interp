# %%
%load_ext autoreload
%autoreload 2
# %%
import gc
import json
import os
import sys
import random
from typing import Literal
sys.path.append("..")  # For imports from parent dir
from utils.distances import build_knn_graph, geodesic_distance_matrix, cosine_similarity_matrix, mp_phylogenetic_distance_matrix
from utils.data import load_data_from_hf, remove_tags
from utils.phylogenetics import get_tag_to_gtdb_accession_map
from utils.sampling import sample_genome
from utils.inference import get_mean_embeddings, batch_tokenize

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
import umap
import datasets
from evo2 import Evo2
# from id2taxonomy import get_taxonomy_from_accession
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

def preprocess(df: pd.DataFrame, min_length: int, subset: int) -> pd.DataFrame:
    """Filter sequences by minimum length, shuffle, and return a subset."""
    print("Preprocessing data...")
    df["sequence_length"] = df["text"].apply(len)
    filtered_df = df[df["sequence_length"] > min_length]
    shuffled_df = filtered_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    shuffled_df = shuffled_df.rename(columns={"text": "sequence"})
    sequences_with_tags_removed, tags = remove_tags(shuffled_df["sequence"])
    shuffled_df["sequence"] = sequences_with_tags_removed
    shuffled_df["tags"] = tags
    shuffled_df["class"] = shuffled_df["tags"].str.split(";").str[2]
    shuffled_df["order"] = shuffled_df["tags"].str.split(";").str[3]
    shuffled_df["family"] = shuffled_df["tags"].str.split(";").str[4]
    return shuffled_df.head(subset)

def add_gtdb_accession(df: pd.DataFrame) -> pd.DataFrame:
    """Add Gtdb accession IDs to the dataframe."""
    print("Adding Gtdb accession IDs...")
    df["gtdb_accession"] = df["tags"].map(get_tag_to_gtdb_accession_map())
    return df

df = preprocess(df, NUM_SAMPLES * REGION_LENGTH, subset=NUM_SPECIES)
df = add_gtdb_accession(df)
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

# inputs_by_species = []

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
def umap_fit_transform(embeddings: torch.Tensor) -> torch.Tensor:
    """Fit UMAP on embeddings and return 3D reduced embeddings."""
    reducer = umap.UMAP(n_components=3, random_state=42)
    umap_embeddings = reducer.fit_transform(
        embeddings.to(torch.float32).cpu().numpy()
    )
    return torch.tensor(umap_embeddings, dtype=torch.float32)

embedding_3d = umap_fit_transform(mean_embeddings)

# %%
def plot_3d_embedding(embedding_3d: torch.Tensor, title: str, labels: list[str]) -> go.Figure:
    """Plot 3D UMAP embedding using Plotly, coloring by categorical label."""
    # Ensure data is NumPy for Plotly rendering
    coords = (
        embedding_3d.detach().cpu().numpy()
        if isinstance(embedding_3d, torch.Tensor)
        else np.asarray(embedding_3d)
    )
    assert coords.shape[1] == 3, "Embedding must be 3D"
    assert len(labels) == coords.shape[0], "Labels length must match number of points"

    # Map labels to colors (cycle palette if needed)
    unique_labels = sorted(set(labels))
    palette = px.colors.qualitative.Light24
    if len(unique_labels) > len(palette):
        repeats = (len(unique_labels) // len(palette)) + 1
        palette = (palette * repeats)[: len(unique_labels)]
    color_map = {label: color for label, color in zip(unique_labels, palette)}
    colors = [color_map[label] for label in labels]

    # Create main trace with points
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    opacity=0.7,
                    color=colors,
                ),
                text=[f"Genome {i}<br>Label: {labels[i]}" for i in range(coords.shape[0])],
                hovertemplate="<b>%{text}</b><br>UMAP 1: %{x}<br>UMAP 2: %{y}<br>UMAP 3: %{z}<extra></extra>",
                showlegend=False,
            )
        ]
    )

    # Add legend-only entries for categories
    for label, color in color_map.items():
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(
                    size=4,
                    opacity=0.7,
                    color=color,
                ),
                name=label,
                showlegend=True,
                visible="legendonly",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
        width=800,
        height=600,
        legend=dict(itemsizing="constant"),
    )
    return fig

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

    fig = plot_3d_embedding(
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
def plot_distance_correlation(x, y, x_label, y_label, title):
    fig = px.scatter(
        x=x,
        y=y,
        labels={"x": x_label, "y": y_label},
        title=title,
        trendline_color_override="red"
    )
    fig.update_traces(marker=dict(size=5, opacity=0.3))
    fig.show()

# Use lower triangular indices, excluding diagonal
n = cos_similarity_matrix.shape[0]
tril_indices = np.tril_indices(n, k=-1)

plot_distance_correlation(
    x=phylo_distance_matrix[tril_indices].to(torch.float32).cpu().numpy(),
    y=cos_similarity_matrix[tril_indices].to(torch.float32).cpu().numpy(),
    # labels=df["order"].tolist()[:subset],
    x_label="Phylogenetic Distance",
    y_label="Cosine Distance",
    title="Cosine Distance vs Phylogenetic Distance"
)
plot_distance_correlation(
    x=phylo_distance_matrix[tril_indices].to(torch.float32).cpu().numpy(),
    y=geo_distance_matrix[tril_indices],
    # labels=df["order"].tolist()[:subset],
    x_label="Phylogenetic Distance",
    y_label="Geodesic Distance",
    title="Geodesic Distance vs Phylogenetic Distance"
)
# %%
