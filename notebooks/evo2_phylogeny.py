# %%
# %load_ext autoreload
# %autoreload 2

# %%
import gc
import os
from pathlib import Path
import sys
import random

sys.path.append("..")  # For imports from parent dir

from utils.config import create_default_config
from utils.distances import (
    build_knn_graph,
    geodesic_distance_matrix,
    cosine_similarity_matrix,
    mp_phylogenetic_distance_matrix,
)
from utils.data import load_data_from_hf, preprocess_gtdb_sequences, add_gtdb_accession
from utils.phylogenetics import get_tag_to_gtdb_accession_map, filter_genomes_in_tree
from utils.sampling import sample_genome
from utils.inference import get_mean_embeddings, batch_tokenize
from utils.visualization import (
    umap_reduce_3d,
    plot_umap_3d,
    plot_distance_scatter,
    save_plotly_figure,
)

import numpy as np
import torch
from evo2 import Evo2
from tqdm import tqdm
import pandas as pd
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Create configuration with custom parameters
config = create_default_config(
    model="7b",
    data_sources=[
        "https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/midtraining_specific/gtdb_v220_stitched/data_gtdb_train_chunk1.jsonl.gz",
        "https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/midtraining_specific/gtdb_v220_stitched/data_gtdb_train_chunk11.jsonl.gz",
        "https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/midtraining_specific/gtdb_v220_stitched/data_gtdb_train_chunk21.jsonl.gz",
    ],
    num_species=2458,
    num_samples=None,
    coverage_fraction=0.05,
    region_length=4000,
    average_over_last_bp=2000,
    layer_name="blocks.24.mlp.l3",
    random_seed=42,
    min_sequence_length=40000,
    remove_tags=True,
)

# Print configuration summary
config.print_summary()

# Save configuration to experiment directory
experiment_dir = config.get_cache_path()
config.save(experiment_dir)

# Enable PyTorch optimizations
random.seed(config.random_seed)
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = True

assert os.getcwd().endswith("evo2-mech-interp/notebooks"), (
    "Run from notebooks/ directory"
)

# %%
# =============================================================================
# DATA LOADING
# =============================================================================

data_files = config.data_sources

# TODO: Switch to parquet at least, or ideally use something better like hf datasets
if not Path(experiment_dir / "genomes_metadata.csv").exists():
    df = load_data_from_hf(data_files)

# %%
# =============================================================================
# PREPROCESSING
# =============================================================================
if not Path(experiment_dir / "genomes_metadata.csv").exists():
    df = preprocess_gtdb_sequences(
        df,
        min_length=config.min_sequence_length,
        subset=config.num_species,
        random_seed=config.random_seed,
        remove_tags=config.remove_tags,
    )
    df = add_gtdb_accession(df, get_tag_to_gtdb_accession_map())
    df = df.dropna(subset=["gtdb_accession"]).reset_index(drop=True)

    # Filter to only keep genomes that are in the phylogenetic tree
    # This prevents wasting resources on genomes that will be dropped later
    df = filter_genomes_in_tree(
        df, config.gtdb_tree_path, accession_column="gtdb_accession"
    )

    # Save metadata to experiment directory
    df.to_csv(experiment_dir / "genomes_metadata.csv", index=False)
    print(f"Saved metadata for {len(df)} genomes")
else:
    df = pd.read_csv(experiment_dir / "genomes_metadata.csv")
    print(f"Loaded metadata for {len(df)} genomes from cache")
df.head()

# %%
# =============================================================================
# SAMPLING
# =============================================================================

samples = {
    "genome_idx": [],
    "sample": [],
}

sampling_kwargs = {
    "sample_region_length": config.region_length,
}
if config.num_samples is not None:
    sampling_kwargs["num_samples"] = config.num_samples
    sampling_kwargs["coverage_fraction"] = None
else:
    sampling_kwargs["coverage_fraction"] = config.coverage_fraction
    sampling_kwargs["num_samples"] = None

if not Path(experiment_dir / "sampled_regions.csv").exists():
    print("Sampling regions from genomes...")
    for row in df.itertuples():
        sampled_regions = sample_genome(row.sequence, **sampling_kwargs)
        samples["genome_idx"].extend([row.Index] * len(sampled_regions))
        samples["sample"].extend(sampled_regions)
        if len(set(samples["genome_idx"])) >= config.num_species:
            break

    samples_df = pd.DataFrame(samples)
    samples_df.to_csv(experiment_dir / "sampled_regions.csv", index=False)
    print(
        f"Sampled {len(samples_df)} regions from {len(samples_df['genome_idx'].unique())} genomes"
    )
else:
    samples_df = pd.read_csv(experiment_dir / "sampled_regions.csv")
    print(
        f"Loaded {len(samples_df)} sampled regions from {len(samples_df['genome_idx'].unique())} genomes from cache"
    )
samples_df.head()

# %%
# =============================================================================
# MODEL LOADING
# =============================================================================

evo2_model = Evo2(config.model_name)
print(
    f"Loaded {config.model_name} with {sum(p.numel() for p in evo2_model.model.parameters() if p.requires_grad):,} parameters"
)

# %%
# =============================================================================
# TOKENIZATION
# =============================================================================

torch.cuda.empty_cache()
gc.collect()

print("Pre-tokenizing all sequences...")
tokenized_samples = []
for genome_idx, genome_df in tqdm(samples_df.groupby("genome_idx"), desc="Tokenizing"):
    tokenized_samples.append(
        batch_tokenize(genome_df["sample"].tolist(), evo2_model, config.batch_size)
    )

# %%
# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

print("Calculating mean embeddings for all genomes...")
mean_embeddings = get_mean_embeddings(
    df=df,
    tokenized_samples=tokenized_samples,
    evo2_model=evo2_model,
    batch_size=config.batch_size,
    experiment_dir=experiment_dir,
    d_model=config.d_model,
    region_length=config.region_length,
    layer_name=config.layer_name,
    average_over_last_bp=config.average_over_last_bp,
    force_recompute=False,  # Set to True to ignore cache
)

print(f"Final embeddings shape: {mean_embeddings.shape}")

# %%
# =============================================================================
# DIMENSIONALITY REDUCTION
# =============================================================================

embedding_3d = umap_reduce_3d(mean_embeddings, random_state=config.random_seed)
print(f"UMAP embeddings shape: {embedding_3d.shape}")

# %%
# =============================================================================
# VISUALIZATION: 3D UMAP BY TAXONOMY
# =============================================================================

for category in ["class", "order", "family"]:
    # Render ALL points; map rare categories to "Other" for readability
    labels_series = df[category].fillna("Unknown").replace("NONE", "Unknown")
    value_counts = labels_series.value_counts()
    min_count = max(2, int(0.02 * len(labels_series)))  # at least 2, or 2%
    frequent = set(value_counts[value_counts >= min_count].index)
    if len(frequent) == 0 and len(value_counts) > 0:
        frequent = set(value_counts.head(min(10, len(value_counts))).index)

    # Align labels with embedding points
    n_points = embedding_3d.shape[0]
    labels_list = [
        lbl if lbl in frequent else "Other" for lbl in labels_series.tolist()
    ]
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

    # Save to HTML and PNG
    filepath = experiment_dir / f"umap_3d_{category}"
    saved_formats = save_plotly_figure(fig, filepath, formats=["html", "png"])
    print(f"Saved {category} visualization: {', '.join(saved_formats)}")

# %%
# =============================================================================
# PHYLOGENETIC DISTANCE MATRIX
# =============================================================================

print("Computing phylogenetic distance matrix...")
if not Path(experiment_dir / "phylogenetic_distance_matrix.pt").exists():
    phylo_distance_matrix = mp_phylogenetic_distance_matrix(
        df["gtdb_accession"].tolist(), config.gtdb_tree_path
    )
    torch.save(
        phylo_distance_matrix, experiment_dir / "phylogenetic_distance_matrix.pt"
    )
else:
    phylo_distance_matrix = torch.load(
        experiment_dir / "phylogenetic_distance_matrix.pt"
    )
print(f"Phylogenetic distance matrix shape: {phylo_distance_matrix.shape}")

# %%
# =============================================================================
# EMBEDDING DISTANCE MATRICES
# =============================================================================

print("Building KNN graph and computing distances...")
adjacency_matrix = build_knn_graph(
    mean_embeddings, k=27, distance="cosine", weighted=True
)
geo_distance_matrix = geodesic_distance_matrix(adjacency_matrix)
cos_similarity_matrix = cosine_similarity_matrix(mean_embeddings)

print(f"Geodesic distance matrix shape: {geo_distance_matrix.shape}")
print(f"Cosine similarity matrix shape: {cos_similarity_matrix.shape}")

# %%
# =============================================================================
# DISTANCE CORRELATION PLOTS
# =============================================================================

# Use lower triangular indices, excluding diagonal
n = cos_similarity_matrix.shape[0]
tril_indices = np.tril_indices(n, k=-1)

# Convert cosine similarity to distance for clearer interpretation
cos_distance_matrix = 1 - cos_similarity_matrix

# Plot cosine distance vs phylogenetic distance with Chatterjee and Spearman
fig_cosine = plot_distance_scatter(
    x=phylo_distance_matrix[tril_indices].to(torch.float32).cpu().numpy(),
    y=cos_distance_matrix[tril_indices].to(torch.float32).cpu().numpy(),
    x_label="Phylogenetic Distance",
    y_label="Cosine Distance",
    title="Cosine Distance vs Phylogenetic Distance",
    include_correlations=["chatterjee", "spearman"],
)
# fig_cosine.show()

# Save cosine distance plot
filepath = experiment_dir / "distance_cosine_vs_phylogenetic"
saved_formats = save_plotly_figure(fig_cosine, filepath, formats=["html", "png"])
print(f"Saved cosine distance plot: {', '.join(saved_formats)}")

# Plot geodesic distance vs phylogenetic distance with Pearson
fig_geodesic = plot_distance_scatter(
    x=phylo_distance_matrix[tril_indices].to(torch.float32).cpu().numpy(),
    y=geo_distance_matrix[tril_indices],
    x_label="Phylogenetic Distance",
    y_label="Geodesic Distance",
    title="Geodesic Distance vs Phylogenetic Distance",
    include_correlations=["pearson"],
)
# fig_geodesic.show()

# Save geodesic distance plot
filepath = experiment_dir / "distance_geodesic_vs_phylogenetic"
saved_formats = save_plotly_figure(fig_geodesic, filepath, formats=["html", "png"])
print(f"Saved geodesic distance plot: {', '.join(saved_formats)}")

# %%
from utils.subspace import FlatSubspace, FlatSubspaceLoss

# Initialize wandb
wandb.init(
    project="evo2-mech-interp",
    name="flat_subspace_training_vectorised",
    config={
        "input_dim": config.d_model,
        "output_dim": 10,
        "alpha": 0.2,
        "lr": 1e-3,
        "n_epochs": 100,
        "n_genomes": mean_embeddings.shape[0],
    },
)

model = FlatSubspace(config.d_model, 10).to(device)
loss_fn = FlatSubspaceLoss(alpha=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Move data to device once
mean_embeddings = mean_embeddings.to(device).to(torch.float32)
phylo_distance_matrix = phylo_distance_matrix.to(device).to(torch.float32)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    # Forward pass: compute all pairwise distances at once
    d_pred_matrix = model(mean_embeddings)  # (2400, 2400)

    # Reconstruction
    z = model.encode(mean_embeddings)
    x_recon = model.decode(z)

    # Compute loss
    total_loss, distance_loss, recon_loss = loss_fn(
        d_pred_matrix, phylo_distance_matrix, x_recon, mean_embeddings
    )

    # Backward pass
    total_loss.backward()
    optimizer.step()

    # Logging
    print(
        f"Epoch {epoch + 1}/100, Loss: {total_loss.item():.4f}, "
        f"Distance: {distance_loss.item():.4f}, Recon: {recon_loss.item():.4f}"
    )

    wandb.log(
        {
            "epoch": epoch,
            "total_loss": total_loss.item(),
            "distance_loss": distance_loss.item(),
            "reconstruction_loss": recon_loss.item(),
            "beta": model.beta.item(),
        }
    )

wandb.finish()

# %%
# =============================================================================
# FLAT SUBSPACE UMAP VISUALIZATION
# =============================================================================

# Extract embeddings from the trained flat subspace
model.eval()
with torch.no_grad():
    flat_subspace_embeddings = model.encode(mean_embeddings).cpu()

print(f"Flat subspace embeddings shape: {flat_subspace_embeddings.shape}")

# Reduce to 3D using UMAP
flat_embedding_3d = umap_reduce_3d(
    flat_subspace_embeddings, random_state=config.random_seed
)
print(f"Flat subspace UMAP embeddings shape: {flat_embedding_3d.shape}")

# %%
# =============================================================================
# VISUALIZATION: 3D UMAP OF FLAT SUBSPACE BY TAXONOMY
# =============================================================================

for category in ["family", "class", "order"]:
    # Render ALL points; map rare categories to "Other" for readability
    labels_series = df[category].fillna("Unknown").replace("NONE", "Unknown")
    value_counts = labels_series.value_counts()
    min_count = max(2, int(0.02 * len(labels_series)))  # at least 2, or 2%
    frequent = set(value_counts[value_counts >= min_count].index)
    if len(frequent) == 0 and len(value_counts) > 0:
        frequent = set(value_counts.head(min(10, len(value_counts))).index)

    # Align labels with embedding points
    n_points = flat_embedding_3d.shape[0]
    labels_list = [
        lbl if lbl in frequent else "Other" for lbl in labels_series.tolist()
    ]
    if len(labels_list) > n_points:
        labels_for_plot = labels_list[:n_points]
    elif len(labels_list) < n_points:
        labels_for_plot = labels_list + ["Other"] * (n_points - len(labels_list))
    else:
        labels_for_plot = labels_list

    fig = plot_umap_3d(
        flat_embedding_3d,
        title=f"3D UMAP of Flat Subspace Colored by {category.capitalize()}",
        labels=labels_for_plot,
    )
    # fig.show()

    # Save to HTML and PNG
    filepath = experiment_dir / f"flat_subspace_umap_3d_{category}"
    saved_formats = save_plotly_figure(fig, filepath, formats=["html", "png"])
    print(f"Saved flat subspace {category} visualization: {', '.join(saved_formats)}")

# %%
# =============================================================================
# FLAT SUBSPACE ANGULAR DISTANCE VS PHYLOGENETIC DISTANCE
# =============================================================================

print("Computing angular distances in flat subspace...")
model.eval()
with torch.no_grad():
    # Compute the full angular distance matrix using the trained model
    angular_distance_matrix = model(mean_embeddings).cpu()

print(f"Angular distance matrix shape: {angular_distance_matrix.shape}")

# Plot angular distance vs phylogenetic distance
fig_angular = plot_distance_scatter(
    x=phylo_distance_matrix[tril_indices].to(torch.float32).cpu().numpy(),
    y=angular_distance_matrix[tril_indices].to(torch.float32).cpu().numpy(),
    x_label="Phylogenetic Distance",
    y_label="Angular Distance (Flat Subspace)",
    title="Angular Distance in Flat Subspace vs Phylogenetic Distance",
    include_correlations=["pearson", "spearman", "chatterjee"],
)
# fig_angular.show()

# Save angular distance plot
filepath = experiment_dir / "distance_angular_vs_phylogenetic"
saved_formats = save_plotly_figure(fig_angular, filepath, formats=["html", "png"])
print(f"Saved angular distance plot: {', '.join(saved_formats)}")

# %%
