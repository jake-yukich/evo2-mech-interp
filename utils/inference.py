
import pandas as pd
import torch

import os
import gc
import hashlib
from itertools import batched

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
    """Run inference  in batches on a list of tokenized samples. Returns concatenated embeddings of shape."""
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
    cache_path: str,
    d_model: int,
    region_length: int,
    layer_name: str,
    average_over_last_bp: int,
):
    print("Processing embeddings...")
    if os.path.exists(f"{cache_path}/all_genomes_embeddings.pt"):
        print("Loading all genomes embeddings from cache...")
        mean_embeddings_tensor = torch.load(f"{cache_path}/all_genomes_embeddings.pt")
        assert mean_embeddings_tensor.shape[1] == d_model, f"Expected: {d_model}, Got: {mean_embeddings_tensor.shape[1]}"
        return mean_embeddings_tensor
    else:
        batch_mean_embeddings = []
        for genome_idx, tokenized_samples_from_genome in enumerate(
            tqdm(tokenized_samples, desc="Processing genomes")
        ):
            sample_embeddings = batch_inference(
                tokenized_samples_from_genome, evo2_model, batch_size, region_length, d_model, layer_name
            )
            genome_hash = hashlib.sha256(df.at[genome_idx, "sequence"].encode()).hexdigest()

            if os.path.exists(f"{cache_path}/genome_{genome_idx}_hash_{genome_hash}.pt"):
                print(f"Skipping genome {genome_idx}, already processed.")
                batch_mean_embeddings.append(torch.load(f"{cache_path}/genome_{genome_idx}_hash_{genome_hash}.pt"))
                continue
            
            genome_embedding = sample_embeddings[:, -average_over_last_bp:, :].mean(
                dim=(0, 1)
            )
            assert genome_embedding.shape == (d_model,)
            
            torch.save(
                genome_embedding, f"{cache_path}/{genome_hash}.pt"
            )
            
            batch_mean_embeddings.append(genome_embedding)
        mean_embeddings = torch.stack(batch_mean_embeddings, dim=0)
        assert mean_embeddings.shape == (
            len(mean_embeddings),
            d_model,
        ), f"Expected: {(len(mean_embeddings), d_model)}, Got: {mean_embeddings.shape}"
        torch.save( 
            mean_embeddings, f"{cache_path}/all_genomes_embeddings.pt"
        )
        return mean_embeddings