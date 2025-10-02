
import pandas as pd
import torch

import os
import gc
import hashlib
from itertools import batched
from pathlib import Path

from tqdm import tqdm
from evo2 import Evo2

def batch_tokenize(
    sequences: list[str], model: Evo2, batch_size: int
) -> list[torch.Tensor]:
    return [model.tokenizer.tokenize(sample) for sample in sequences]

@torch.no_grad()
def batch_inference(
    tokenized_samples: list[torch.Tensor],
    model: Evo2,
    batch_size: int,
    region_length: int,
    d_model: int,
    layer_name: str,
) -> torch.Tensor:
    """Run inference in batches on a list of tokenized samples. Returns concatenated embeddings of shape."""
    sample_embeddings = []
    for batch in batched(tokenized_samples, batch_size):
        input_ids = torch.tensor(batch, dtype=torch.int).to("cuda:0")
        input_ids = input_ids.unsqueeze(0) if input_ids.ndim == 1 else input_ids
        assert input_ids.shape == (len(batch), region_length), (
            f"{input_ids.shape = }"
        )

        _, embeddings = model(
            input_ids, return_embeddings=True, layer_names=[layer_name]
        )

        sample_embeddings.append(
            embeddings[layer_name].cpu()
        )  # Move to CPU immediately to save GPU memory

    # Clear GPU cache periodically
    if len(sample_embeddings) % 10 == 0:
        torch.cuda.empty_cache()

    sample_embeddings = torch.cat(sample_embeddings, dim=0)
    assert sample_embeddings.shape == (
            len(tokenized_samples),
            region_length,
            d_model,
        ), f"Expected: {(len(tokenized_samples), region_length, d_model)}, Got: {sample_embeddings.shape}"
    return sample_embeddings


@torch.no_grad()
def get_mean_embeddings(
    df: pd.DataFrame,
    tokenized_samples: list[list[torch.Tensor]],
    evo2_model: Evo2,
    batch_size: int,
    experiment_dir: Path,
    d_model: int,
    region_length: int,
    layer_name: str,
    average_over_last_bp: int,
    force_recompute: bool = False,
) -> torch.Tensor:
    """Compute or load mean embeddings for all genomes."""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create embeddings subdirectory for individual genome embeddings
    embeddings_dir = experiment_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    
    # Aggregated embeddings are stored at experiment root
    all_embeddings_file = experiment_dir / "all_genomes_embeddings.pt"
    
    # Check if we can load from cache
    if not force_recompute and all_embeddings_file.exists():
        print("Loading all genomes embeddings from cache...")
        mean_embeddings_tensor = torch.load(all_embeddings_file)
        
        # Validate shape
        if mean_embeddings_tensor.shape[1] != d_model:
            print(f"Warning: Cached embeddings have wrong dimension ({mean_embeddings_tensor.shape[1]} != {d_model})")
            print("Recomputing embeddings...")
        elif mean_embeddings_tensor.shape[0] != len(df):
            print(f"Warning: Cached embeddings have wrong number of genomes ({mean_embeddings_tensor.shape[0]} != {len(df)})")
            print("Recomputing embeddings...")
        else:
            print(f"Successfully loaded {mean_embeddings_tensor.shape[0]} genome embeddings from cache")
            return mean_embeddings_tensor
    
    # Compute embeddings
    print("Computing embeddings for all genomes...")
    batch_mean_embeddings = []
    
    for genome_idx, tokenized_samples_from_genome in enumerate(
        tqdm(tokenized_samples, desc="Processing genomes")
    ):
        # Generate hash for this specific genome sequence
        genome_hash = hashlib.sha256(df.at[genome_idx, "sequence"].encode()).hexdigest()
        genome_cache_file = embeddings_dir / f"{genome_hash}.pt"
        
        # Try to load from per-genome cache
        if not force_recompute and genome_cache_file.exists():
            try:
                genome_embedding = torch.load(genome_cache_file)
                if genome_embedding.shape == (d_model,):
                    batch_mean_embeddings.append(genome_embedding)
                    continue
                else:
                    print(f"Warning: Cached genome {genome_idx} has wrong shape, recomputing...")
            except Exception as e:
                print(f"Warning: Error loading cached genome {genome_idx}: {e}, recomputing...")
        
        # Compute embedding for this genome
        sample_embeddings = batch_inference(
            tokenized_samples_from_genome, evo2_model, batch_size, 
            region_length, d_model, layer_name
        )
        
        genome_embedding = sample_embeddings[:, -average_over_last_bp:, :].mean(dim=(0, 1))
        assert genome_embedding.shape == (d_model,), f"Expected shape ({d_model},), got {genome_embedding.shape}"
        
        # Cache this genome's embedding in the embeddings subdirectory
        torch.save(genome_embedding, genome_cache_file)
        batch_mean_embeddings.append(genome_embedding)
    
    # Stack all embeddings
    mean_embeddings = torch.stack(batch_mean_embeddings, dim=0)
    assert mean_embeddings.shape == (len(df), d_model), \
        f"Expected: {(len(df), d_model)}, Got: {mean_embeddings.shape}"
    
    # Save aggregated embeddings at experiment root
    torch.save(mean_embeddings, all_embeddings_file)
    print(f"Saved {mean_embeddings.shape[0]} genome embeddings to {all_embeddings_file}")
    
    return mean_embeddings