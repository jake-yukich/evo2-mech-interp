# Copilot Instructions for evo2-mech-interp

## Project Overview
This is a mechanistic interpretability project analyzing genomic foundation models (Evo2). The core workflow extracts embeddings from bacterial genome sequences, compares them to phylogenetic relationships from the GTDB tree, and visualizes correlation patterns using distance metrics.

## Architecture & Data Flow

### Core Pipeline (see `notebooks/phylo_tree_prod.py`)
1. **Data Loading**: Download GTDB bacterial sequences from HuggingFace (`arcinstitute/opengenome2`)
2. **Sampling**: Extract non-overlapping genomic regions from each genome (configurable by `NUM_SAMPLES` or `COVERAGE_FRACTION`)
3. **Tree Validation**: Filter genomes to keep only those with accession IDs present in the GTDB phylogenetic tree (prevents wasting resources)
4. **Inference**: Generate embeddings via Evo2 model (1B or 7B) for specific layers (e.g., `blocks.24.mlp.l3`)
5. **Aggregation**: Average embeddings over last N base pairs of each region, then across regions per genome
6. **Distance Analysis**: Compare cosine/geodesic distances in embedding space vs phylogenetic tree distances
7. **Visualization**: UMAP 3D projections colored by taxonomic levels (class/order/family)

### Key Directories
- `utils/`: Modular functions split by concern (data, distances, inference, phylogenetics, sampling)
- `data/embeddings/model_{1b|7b}/{sampling_config}/layer_{name}/`: Hierarchical caching of embeddings by model, sampling strategy, and layer
- `notebooks/`: Exploratory scripts and production pipelines (`.py` files are converted from Jupyter notebooks)

## Critical Patterns

### Caching Strategy
- **Per-genome caching**: Individual embeddings saved as `{sha256_hash}.pt` to enable incremental computation
- **Aggregate caching**: `all_genomes_embeddings.pt` stores final mean embeddings across all genomes
- **Metadata caching**: `genomes_metadata.csv` and `sampled_regions.csv` track provenance
- Always check `os.path.exists()` before recomputing expensive embeddings

### Tag Processing
- Genomic sequences contain embedded phylogenetic tags in format `|D__BACTERIA;P__...;C__...;O__...;F__...;G__...;S__...|`
- Use `utils.data.remove_tags()` to extract tags and clean sequences separately
- Map tags to GTDB accessions via `utils.phylogenetics.get_tag_to_gtdb_accession_map()`

### Tree Validation
- **IMPORTANT**: Always filter genomes early to ensure accession IDs are in the phylogenetic tree
- Use `utils.phylogenetics.filter_genomes_in_tree()` AFTER adding GTDB accessions but BEFORE sampling/inference
- This prevents wasting GPU resources on genomes that will be dropped later when computing phylogenetic distances
- The `mp_phylogenetic_distance_matrix()` function returns NaN for accession IDs not in the tree, which causes issues downstream

### GPU Memory Management
```python
# Pattern used throughout for memory efficiency:
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()  # Call periodically in loops

# Move embeddings to CPU immediately after extraction:
embeddings[layer_name].cpu()
```

### Distance Metrics
- **Cosine distance**: Used for embedding space comparisons (convert similarity to distance: `1 - cosine_sim`)
- **Geodesic distance**: Computed on KNN graph (k=27 typical) using Dijkstra's algorithm (`scipy.sparse.csgraph`)
- **Phylogenetic distance**: Tree distances from GTDB bacterial tree (`data/gtdb/bac120_r220.tree`) using `ete3`

### Multiprocessing for Tree Distances
- Phylogenetic distance matrix computation is parallelized (default 12 processes)
- Each worker loads its own `Tree` instance (not picklable)
- See `utils.distances.mp_phylogenetic_distance_matrix()`

## Configuration Constants

Located at top of scripts (e.g., `phylo_tree_prod.py`):
- `MODEL`: "1b" or "7b" (determines `D_MODEL` dimensions: 1920 or 4096)
- `NUM_SPECIES`: Number of genomes to sample
- `NUM_SAMPLES` / `COVERAGE_FRACTION`: Sampling strategy (mutually exclusive)
- `REGION_LENGTH`: Base pairs per genomic sample (typically 4000)
- `AVERAGE_OVER_LAST_BP`: Only average activations over last N bp (typically 2000) to avoid initial noise
- `LAYER_NAME`: Target layer for extraction (e.g., `"blocks.24.mlp.l3"`)
- `BATCH_SIZE`: 48 for 1B model, 8 for 7B model

## Development Workflows

### Running Experiments
```bash
# From project root:
cd notebooks
python phylo_tree_prod.py

# Expects:
# - data/gtdb/bac120_r220.tree (download GTDB tree)
# - data/gtdb/bac120_metadata_r220.tsv (download GTDB metadata)
# - ../utils/ module accessible (uses sys.path.append(".."))
```

### Module Imports
Scripts use relative imports from project root:
```python
sys.path.append("..")  # Common in notebooks/
from utils.distances import build_knn_graph, geodesic_distance_matrix
from utils.phylogenetics import get_tag_to_gtdb_accession_map
```

### Data Access
- GTDB sequences loaded via `datasets.load_dataset("json", data_files=[url])` 
- Tree file at `/root/evo2-mech-interp/data/gtdb/bac120_r220.tree`
- Format: `Tree(path, format=1, quoted_node_names=True)` for ete3

## Project-Specific Gotchas

1. **Notebook-style code**: Scripts contain `# %%` cell separators but are `.py` files (converted from Jupyter)
2. **Hardcoded paths**: Absolute paths like `/root/evo2-mech-interp/` appear in scripts; adjust for environment
3. **No tests**: Project has no test suite; validation is visual (plotly scatter plots)
4. **TODOs**: Several refactoring notes marked with `# TODO:` (e.g., sampling bias, tree downloading)
5. **Random seed**: Set to 42 globally but sampling uses `random.randint()` after seed
6. **Accession ID format**: GTDB uses quoted node names in tree; ensure `quoted_node_names=True` when loading

## Dependencies

Core libraries (see `pyproject.toml`):
- `torch>=2.8.0` (PyTorch for embeddings)
- `evo2>=0.3.0` (Genomic foundation model)
- `datasets>=2.14.0` (HuggingFace datasets)
- `ete3` (phylogenetic tree manipulation)
- `umap-learn`, `plotly` (visualization)
- `scipy` (sparse graphs, Dijkstra)

## Visualization Conventions

- 3D UMAP plots colored by taxonomic rank (class/order/family)
- Rare categories (<5% of samples) grouped as "Other"
- Scatter plots compare phylogenetic distance (x-axis) vs embedding distance (y-axis)
- Use `plotly` for interactive 3D: `go.Figure()` with `go.Scatter3d()`
