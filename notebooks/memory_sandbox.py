# %%
import gc
import os
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
from utils.visualization import umap_reduce_3d, plot_umap_3d, plot_distance_scatter

import numpy as np
import torch
from evo2 import Evo2
from tqdm import tqdm
import pandas as pd
import contextlib
from pathlib import Path

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tuning knobs
batch_size = 1
seq_len = 1024
vocab_size = 1000
dtype_choice = "float8"
use_amp = True

# Profiler schedule
wait_steps = 1
warmup_steps = 1
active_steps = 3

layer_names = ["blocks.24.mlp.l3"]
log_dir = Path("./_profiler_logs/memory_sandbox")

# %%
model = Evo2("evo2_7b")
model.model.eval()

# %%
def make_input_ids(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    sequences = ["".join(random.choice("ACGT") for _ in range(seq_len)) for _ in range(batch_size)]
    tokenized = [model.tokenizer.tokenize(seq) for seq in sequences]
    tensors = [torch.tensor(t, dtype=torch.long, device=device) for t in tokenized]
    for t in tensors:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != seq_len:
            raise ValueError(f"Tokenized length {t.shape[0]} != seq_len {seq_len}. Adjust seq_len to tokenizer output.")
    input_ids = torch.stack(tensors, dim=0)
    return input_ids


def get_autocast_context(dtype_name: str, device: torch.device):
    if device.type == "cuda" and use_amp and dtype_name in ("float16", "bfloat16"):
        return torch.autocast(device_type="cuda", dtype=getattr(torch, dtype_name))
    return contextlib.nullcontext()


def profile_embeddings() -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    input_ids = make_input_ids(batch_size, seq_len, vocab_size, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    schedule = torch.profiler.schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1)
    handler = torch.profiler.tensorboard_trace_handler(str(log_dir), worker_name="memory_sandbox")

    with torch.inference_mode():
        with get_autocast_context(dtype_choice, device):
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            with torch.profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                total_steps = wait_steps + warmup_steps + active_steps
                for _ in range(total_steps):
                    _, embeddings = model(input_ids, return_embeddings=True, layer_names=layer_names)
                    # ensure work is finished before stepping the profiler
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    prof.step()
                    del embeddings

    # Top ops by memory usage
    try:
        sort_key = "self_cuda_memory_usage" if device.type == "cuda" else "self_cpu_memory_usage"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
    except Exception as e:
        print(f"Profiler summary unavailable: {e}")

    # CPU memory table
    try:
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))
    except Exception:
        pass

    # CUDA memory stats
    if device.type == "cuda":
        max_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(f"CUDA max allocated: {max_alloc:.1f} MiB, max reserved: {max_reserved:.1f} MiB")
        print(f"CUDA mem free/total: {free_bytes / 1024 ** 2:.0f}/{total_bytes / 1024 ** 2:.0f} MiB")


# %%
profile_embeddings()
# %%
