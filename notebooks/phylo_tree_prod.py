# %%
import gc
import os
import random
from itertools import batched, product
from pprint import pprint
from time import perf_counter

import plotly.graph_objects as go
import torch
import umap
import datasets
from evo2 import Evo2
# from id2taxonomy import get_taxonomy_from_accession
from tqdm import tqdm
import pandas as pd
import hashlib

random.seed(42)
# Enable optimizations
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
if hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = (
        True  # Use TF32 for faster matmul on A100/H100
    )


NUM_SPECIES = 1024  # Number of species to sample (for demo; set higher for real use)
NUM_SAMPLES = 10  # Number of samples per genome
COVERAGE_FRACTION = 0.05  # Fraction of genome to cover with samples
REGION_LENGTH = 4000  # Length of each genomic region to sample (bp)
AVERAGE_OVER_LAST_BP = (
    2000  # Only average activations over the last N bp of each region
)
D_MODEL_7B = 4096
D_MODEL_1B = 1920
MODEL = "1b"  # "1b" or "7b"
D_MODEL = D_MODEL_1B if MODEL == "1b" else D_MODEL_7B if "7b" else None
BATCH_SIZE = 48 if MODEL == "1b" else 8  # Start here and increase if possible
RANDOM_SEED = 42
LAYER_NAME = "blocks.24.mlp.l3"  # Move outside loop
# CACHE_PATH = f"data/embeddings/model_{MODEL}/num_samples_{NUM_SAMPLES}/layer_{LAYER_NAME}/"
CACHE_PATH = f"data/embeddings/model_{MODEL}/coverage_frac_{COVERAGE_FRACTION}/layer_{LAYER_NAME}/"

os.makedirs("data/embeddings", exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)


# %%
data_files = [
    "https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/pretraining_or_both_phases/gtdb_v220_imgpr/data_gtdb_train_chunk1.jsonl.gz"
]

def load_data_from_hf(data_files: list[str]) -> pd.DataFrame:
    """
    Load dataset from Hugging Face Hub or local files.
    """
    dataset = datasets.load_dataset("json", data_files=data_files)
    return pd.DataFrame(dataset["train"])


df = load_data_from_hf(data_files)

# %% - Get items with longest sequences
# TODO: consider how sampling might be biased by taking top n sorted sequences

def preprocess(df: pd.DataFrame, min_length: int, subset: int) -> pd.DataFrame:
    """Filter sequences by minimum length, shuffle and return a subset."""
    df["sequence_length"] = df["text"].apply(len)
    filtered_df = df[df["sequence_length"] > min_length]
    shuffled_df = filtered_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    shuffled_df = shuffled_df.rename(columns={"text": "sequence"})
    return shuffled_df.head(subset)

df = preprocess(df, NUM_SAMPLES * REGION_LENGTH, subset=NUM_SPECIES)
df.to_csv(f"{CACHE_PATH}/genomes_metadata.csv", index=False)
df.head()

# %%  - SAMPLING
def overlaps(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    """Helper, check if the interval (start, end) overlaps with any in intervals."""
    for used_start, used_end in intervals:
        if not (end <= used_start or start >= used_end):
            return True
    return False

# TODO: shrink this function 
def sample_genome(
    text: str,
    sample_region_length: int,
    coverage_fraction: float | None = None,
    num_samples: int | None = None,
    max_attempts: int = 10000,
) -> list[str]:
    """
    Randomly sample non-overlapping regions from the input text until at least
    `coverage_fraction` (default 5%) of the text is covered OR `num_samples` (default 10) samples are obtained
    .
    Returns a list of non-overlapping sampled regions.
    """
    if (coverage_fraction is not None) and (num_samples is not None):
        raise ValueError("Specify either coverage_fraction or num_samples, not both.")
    if (coverage_fraction is None) and (num_samples is None):
        raise ValueError("Specify either coverage_fraction or num_samples.")
    text_length = len(text)

    if text_length < sample_region_length:
        print(
            f"Warning: Text is shorter than sample_region_length ({text_length} < {sample_region_length})"
        )
        return []

    target_coverage = (
        int(text_length * coverage_fraction)
        if coverage_fraction is not None
        else num_samples * sample_region_length
    )
    covered_bp = 0
    used_intervals = []
    regions = []
    attempts = 0

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
        print(
            f"Warning: Only able to cover {covered_bp} bp out of requested {target_coverage} bp ({coverage_fraction * 100:.1f}% of text)."
        )

    return regions


# Sample and save to new df where each row is a sampled region, referencing original genome
samples = {
    "genome_idx": [],
    "sample": [],
}
for row in df.itertuples():
    sampled_regions = sample_genome(
        row.sequence, 
        sample_region_length=REGION_LENGTH, 
        # num_samples=NUM_SAMPLES
        coverage_fraction=COVERAGE_FRACTION
    )
    samples["genome_idx"].extend([row.Index] * len(sampled_regions))
    samples["sample"].extend(sampled_regions)
    if len(samples) >= NUM_SPECIES:
        break

samples_df = pd.DataFrame(samples)
samples_df.to_csv(f"{CACHE_PATH}/sampled_regions.csv", index=False)
samples_df.head()

# %% - Load model
evo2_model = Evo2("evo2_1b_base") if MODEL == "1b" else Evo2("evo2_7b_base")

# %%
torch.cuda.empty_cache()
gc.collect()

def batch_tokenize(
    sequences: list[str], model: Evo2, batch_size: int
) -> list[torch.Tensor]:
    return [model.tokenizer.tokenize(sample) for sample in sequences]

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

@torch.no_grad()
def batch_inference(
    tokenized_samples: list[torch.Tensor],
    model: Evo2,
    batch_size: int,
) -> torch.Tensor:
    """Run inference  in batches on a list of tokenized samples. Returns concatenated embeddings of shape."""
    sample_embeddings = []
    for batch in batched(tokenized_samples, batch_size):
        input_ids = torch.tensor(batch, dtype=torch.int).to("cuda:0")
        input_ids = input_ids.unsqueeze(0) if input_ids.ndim == 1 else input_ids
        assert input_ids.shape == (len(batch), REGION_LENGTH), (
            f"{input_ids.shape = }"
        )

        _, embeddings = model(
            input_ids, return_embeddings=True, layer_names=[LAYER_NAME]
        )

        sample_embeddings.append(
            embeddings[LAYER_NAME].cpu()
        )  # Move to CPU immediately to save GPU memory

    # Clear GPU cache periodically
    if len(sample_embeddings) % 10 == 0:
        torch.cuda.empty_cache()

    sample_embeddings = torch.cat(sample_embeddings, dim=0)
    assert sample_embeddings.shape == (
            len(tokenized_samples),
            REGION_LENGTH,
            D_MODEL,
        ), f"Expected: {(len(tokenized_samples), REGION_LENGTH, D_MODEL)}, Got: {sample_embeddings.shape}"
    return sample_embeddings

print("Processing embeddings...")
with torch.no_grad():
    mean_embeddings = []
    for genome_idx, tokenized_samples_from_genome in enumerate(
        tqdm(tokenized_samples, desc="Processing genomes")
    ):
        sample_embeddings = batch_inference(
            tokenized_samples_from_genome, evo2_model, BATCH_SIZE
        )
        genome_hash = hashlib.sha256(df.at[genome_idx, "sequence"].encode()).hexdigest()

        if os.path.exists(f"{CACHE_PATH}/genome_{genome_idx}_hash_{genome_hash}.pt"):
            print(f"Skipping genome {genome_idx}, already processed.")
            continue
        
        genome_embedding = sample_embeddings[:, -AVERAGE_OVER_LAST_BP:, :].mean(
            dim=(0, 1)
        )
        assert genome_embedding.shape == (D_MODEL,)
        
        torch.save(
            genome_embedding, f"{CACHE_PATH}/{genome_hash}.pt"
        )
        
        mean_embeddings.append(genome_embedding)
    mean_embeddings_tensor = torch.stack(mean_embeddings, dim=0)
    assert mean_embeddings_tensor.shape == (
        len(mean_embeddings),
        D_MODEL,
    ), f"Expected: {(len(mean_embeddings), D_MODEL)}, Got: {mean_embeddings_tensor.shape}"
    if not os.path.exists(f"{CACHE_PATH}/all_genomes_embeddings.pt"):
        torch.save(
            mean_embeddings_tensor, f"{CACHE_PATH}/all_genomes_embeddings.pt"
        )

# %%
def umap_fit_transform(embeddings: torch.Tensor) -> torch.Tensor:
    """Fit UMAP on embeddings and return 3D reduced embeddings."""
    reducer = umap.UMAP(n_components=3, random_state=42)
    umap_embeddings = reducer.fit_transform(
        embeddings.to(torch.float32).cpu().numpy()
    )
    return torch.tensor(umap_embeddings, dtype=torch.float32)

embedding_3d = umap_fit_transform(mean_embeddings_tensor)

# %%

def plot_3d_embedding(embedding_3d: torch.Tensor, labels: list[str]) -> go.Figure:
    """Plot 3D UMAP embedding using Plotly."""
    assert embedding_3d.shape[1] == 3, "Embedding must be 3D"
    assert len(labels) == embedding_3d.shape[0], "Labels length must match number of points"
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=embedding_3d[:, 0],
                y=embedding_3d[:, 1],
                z=embedding_3d[:, 2],
                mode="markers",
                marker=dict(size=8, color="blue", opacity=0.7),
                text=[f"Genome {i}" for i in range(len(embedding_3d))],
                hovertemplate="<b>%{text}</b><br>UMAP 1: %{x}<br>UMAP 2: %{y}<br>UMAP 3: %{z}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="3D UMAP of Genome Embeddings",
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
        width=800,
        height=600,
    )
    return fig

fig = plot_3d_embedding(embedding_3d, labels=[f"Genome {i}" for i in range(len(embedding_3d))])
fig.show()
